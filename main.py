import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM 
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
import os
import sys
import wandb
import timeit
from Nat_init import *

project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'TGX'))
sys.path.insert(0, os.path.join(project_root, 'NAT'))


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, K=1):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(in_channels=node_feat_dim, 
                                out_channels=hidden_dim, 
                                K=K,) #K is the Chebyshev filter size
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight, h, c):
        r"""
        forward function for the model, 
        this is used for each snapshot
        h: node hidden state matrix from previous time
        c: cell state matrix from previous time
        """

        #x - (num_nodes, node_feat_dim)- nodes features?
        # edge_index - (2, num_edges) - the first row contains source nodes, and the second row contains target nodes
        # edge_weight - represents the weight of each edge in the graph - (num_edges,)
        #h - (num_nodes, hidden_dim)
        #c - (num_nodes, hidden_dim)
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins: # ensures that model weights are reinitialized when necessary.
            lin.reset_parameters()

    def forward(self, x_i, x_j): #pairs of nodes 
        x = x_i * x_j #commonly used in link prediction to capture interaction features.
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x) #probability between 0 and 1, indicating the likelihood that an edge exists.



def test_tgb(h,
             h_0,
             c_0, 
             test_loader, 
             test_snapshots, 
             ts_list,
             node_feat,
             model, 
             link_pred,
             neg_sampler,
             evaluator,
             metric, 
             split_mode='val'):
    
    model.eval()
    link_pred.eval()

    perf_list = []
    ts_idx = min(list(ts_list.keys()))
    max_ts_idx = max(list(ts_list.keys()))

    for batch in test_loader:
        pos_src, pos_dst, pos_t, pos_msg = (
        batch.src,
        batch.dst,
        batch.t,
        batch.msg,
        )
        #"query_batch" - For each positive edge in the `pos_batch`, return a list of negative edges
       # `split_mode` specifies whether the valiation or test evaluation set should be retrieved.
       # modify now to include edge type argument
        neg_batch_list = neg_sampler.query_batch(np.array(pos_src.cpu()), np.array(pos_dst.cpu()), np.array(pos_t.cpu()), split_mode=split_mode)
        
        #^^a list of list; each internal list contains the set of negative edges that
                       # should be evaluated against each positive edge.
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = torch.full((1 + len(neg_batch),), pos_src[idx], device=args.device)
            query_dst = torch.tensor(
                        np.concatenate(
                            ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                            axis=0,
                        ),
                        device=args.device,
                    )
            with torch.no_grad():
                y_pred = link_pred(h[query_src], h[query_dst])
            y_pred = y_pred.squeeze(dim=-1).detach()

            input_dict = {
            "y_pred_pos": np.array([y_pred[0].cpu()]),
            "y_pred_neg": np.array(y_pred[1:].cpu()),
            "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
        
        #* update the model now if the prediction batch has moved to next snapshot
        while (pos_t[-1] > ts_list[ts_idx] and ts_idx < max_ts_idx):
            with torch.no_grad():
                cur_index = test_snapshots[ts_idx]
                cur_index = cur_index.long().to(args.device)
                edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
                h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0)
                h = h.detach()
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            ts_idx += 1

    #* update to the final snapshot
    with torch.no_grad():
        cur_index = test_snapshots[max_ts_idx]
        cur_index = cur_index.long().to(args.device)
        edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
        h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0)
        h = h.detach()
        h_0 = h_0.detach()
        c_0 = c_0.detach()

    test_metrics = float(np.mean(np.array(perf_list)))

    return test_metrics, h, h_0, c_0



    
def create_edges_features(extracted_src, extracted_dst, extracted_features, prev_index):
    # Make canonical order of edges
    canonical_edges = torch.stack([torch.min(extracted_src, extracted_dst), 
           torch.max(extracted_src, extracted_dst)], dim=1)
    unique_edges, inverse_indices = torch.unique(canonical_edges, dim=0, return_inverse=True)
    
    feature_dim = extracted_features.shape[1]
    num_unique_edges = unique_edges.shape[0]
    aggregated_features = torch.zeros((num_unique_edges, feature_dim), device=extracted_features.device)
    aggregated_features = aggregated_features.scatter_add(0, 
        inverse_indices.unsqueeze(1).expand(-1, feature_dim), 
        extracted_features
    )
 
    counts = torch.zeros(num_unique_edges, device=extracted_features.device)
    for i in range(num_unique_edges):
        counts[i] = (inverse_indices == i).sum()
    
    #extract average fetures of edges
    aggregated_features = aggregated_features / counts.unsqueeze(1)

    # for i in range(num_unique_edges):
    #     idxs = (inverse_indices == i).nonzero(as_tuple=True)[0]
    #     if idxs.numel() == 2:
    #         print(f"Unique edge {i}: {unique_edges[i]} aggregated from events at indices {idxs.tolist()}")
    #         print("Extracted features for these events:")
    #         print(extracted_features[idxs])
    #         print("Averaged feature:")
    #         print(aggregated_features[i])
    #         print("-----")
    # print("==========================================")
    
    edge_mapping = []
    for src_val, dst_val in zip(prev_index[0].tolist(), prev_index[1].tolist()):
        #find matching from the snapshot base 
        cond1 = (unique_edges[:, 0] == src_val) & (unique_edges[:, 1] == dst_val)
        cond2 = (unique_edges[:, 0] == dst_val) & (unique_edges[:, 1] == src_val)
        match = cond1 | cond2
        # If a match exist
        indices = torch.nonzero(match, as_tuple=True)[0]
        if indices.numel() > 0:
            idx = indices[0].item()
            edge_mapping.append(idx)
        else:
            print(f"Warning: No matching unique edge")
    
    edge_mapping = torch.tensor(edge_mapping, device=unique_edges.device)
    
    #using edge_mapping to get the aggregated features corresponding to each edge in prev_index:
    edge_attr = aggregated_features[edge_mapping]
  
    return edge_attr
    





if __name__ == '__main__':

    import TGX
    from configs import get_args
    from UTG.utils.utils_func import set_random
    from UTG.utils.data_util import loader
    from NAT.utils import RandEdgeSampler


    args, argsv = get_args()
    set_random(args.seed)

    batch_size = args.batch_size

    #ctdg dataset ->> look at  tgb.linkproppred.dataset_pyg or dataset
    dataset = PyGLinkPropPredDataset(name=args.dataset, root="datasets") #time step,source node,target node,weight of nodes as seen in the csv
    full_data = dataset.get_TemporalData() #defined as src, dst, t, msg between nodes- features!, y - edge label , edge type (optional) , weights .  y is the label indicating if an edge is a true edge, always 1 for true edges
    full_data = full_data.to(args.device) #TemporalData(src=[4873540], dst=[4873540], t=[4873540], msg=[4873540, 1], y=[4873540]) --> no w, no type!! to get data aboput those attributes just  .src[index]
    #get masks
    train_mask = dataset.train_mask #True, False.... -> serves to identify which samples in your dataset should be used for training the model
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    train_edges = full_data[train_mask]  #TemporalData(src=[3413837], dst=[3413837], t=[3413837], msg=[3413837, 1], y=[3413837])
    val_edges = full_data[val_mask]
    test_edges = full_data[test_mask]

    #* set up TGB queries, this is only for val and test
    metric = dataset.eval_metric #mrr ->the evaluation metric for this data
    neg_sampler = dataset.negative_sampler #sampler object -->For each positive edge in the `pos_batch`, return a list of negative edges `split_mode` specifies whether the valiation or test evaluation set should be retrieved. 
    evaluator = Evaluator(name=args.dataset) #evaluation class
    min_dst_idx, max_dst_idx = int(full_data.dst.min()), int(full_data.dst.max()) #dst - A list of destination nodes for the events with shape [num_events]

    print("print data properties in the 0 link", full_data.src [0],full_data.dst[0],full_data.t[0],
                full_data.msg[0],
                full_data.y[0])

    #! set up node features
    node_feat = dataset.node_feat #NONE ---> node features of the dataset with dim [N, feat_dim] , uniq_nodes!!
    if (node_feat is not None):
        print("Node not none")
        node_feat = node_feat.to(args.device)
        node_feat_dim = node_feat.size(1)
    else:
        print("Node is none")
        node_feat_dim = 172 # was 256 in UTG, in NAT is 172
        node_feat = torch.randn((full_data.num_nodes,node_feat_dim)).to(args.device)

    e_feat = full_data.msg
    edge_feat_dim = e_feat.shape[1]
    
    NAT_module = init_nat_module(full_data, e_feat, node_feat, train_edges,val_edges, test_edges, args, argsv)
    random_sampler_train = RandEdgeSampler((train_edges.src, ), (train_edges.dst, ))

    
#     min_dst_idx 0
# max_dst_idx 352621
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="utg",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "gclstm",
            "dataset": args.dataset,
            "time granularity": args.time_scale,
            }
        )
    
    hidden_dim = 256
    
    #* load the discretized version
    data = loader(dataset=args.dataset, time_scale=args.time_scale) #loaded adges - 4873540
    #Number of unique edges:4730223

    train_data = data['train_data'] #each is a dictionary containing:
    # 'edge_index': pos_undirected_edges,
    # total number of nodes across all split; this is the same value for each split -> dict, keys are integer timestamps, values are connection between two nodes src and trg
    # data['edge_index_list'][snapshot_idx]     
    #'num_nodes': num_nodes, -> int number of nodes in the graph
         #'time_length': len(pos_undirected_edges), -> length of edges int
        #'ts_map': ts_map, -> dist, int[0-time_length]: unix, element are the corresponding unix timestamp of the snapshots
        #'original_edges': edge_index_list, #just original list -< not having that!!

    #num nodes 352637, time_length 207, ts maps 207 to unix, "edge index" maps 207 to edge indexes --> how can we know the nodes which are participating in the edge?
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    num_epochs = args.max_epoch
    lr = args.lr


    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print (f"Run {seed}")
        
        #* initialization of the model to prep for training
        model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=1).to(args.device)
        node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)
        link_pred = LinkPredictor(hidden_dim, hidden_dim, 1,
                                2, 0.2).to(args.device)


        optimizer = torch.optim.Adam(
            set(model.parameters()) | set(link_pred.parameters()), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val = 0
        best_test = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            print ("------------------------------------------")
            train_start_time = timeit.default_timer()
            optimizer.zero_grad()
            total_loss = 0
            model.train()
            link_pred.train()
            snapshot_list = train_data['edge_index'] #0: snap, 1: snap.....207:snap
            # print("time", train_data['ts_map']) #{0: 0, 1: 3600, 2: 7200, 3: 10800...207:1861200}
            
            h_0, c_0, h = None, None, None
            total_loss = 0
            for snapshot_idx in range(train_data['time_length']): #207

                optimizer.zero_grad()
                if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                    cur_index = snapshot_list[snapshot_idx] #edge indexes
                    cur_index = cur_index.long().to(args.device)
                    # TODO, also need to support edge attributes correctly in TGX
                    if ('edge_attr' not in train_data):
                        edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
            
                        #masking the right edges by timestemps
                        shot_edge_mask = train_edges.t <= train_data['ts_map'][snapshot_idx]
                        
                        shot_edge_idx = torch.nonzero(shot_edge_mask, as_tuple=True)[0]
                        
                        shot_edge_times = train_edges.t[shot_edge_idx]
                        extracted_src = train_edges.src[shot_edge_idx]
                        extracted_dst = train_edges.dst[shot_edge_idx]
                        extracted_features = train_edges.msg[shot_edge_idx]

                        edges = torch.stack([extracted_src, extracted_dst], dim=1)
                        # print("edges", edges)
                        
                        unique_edges = torch.unique(edges, dim=0)
                       
                        unique_extracted_src = unique_edges[:, 0]
                        unique_extracted_dst = unique_edges[:, 1]

                        merged_extracted = torch.cat([unique_extracted_src, unique_extracted_dst])
                        
                        # Compare with cur_index (which should be in the same order)
                        assert torch.equal(unique_extracted_src, cur_index[0][:len(unique_extracted_src)]), "Source nodes do not match!" 
                        assert torch.equal(unique_extracted_dst, cur_index[1][:len(unique_extracted_dst)]), "Target nodes do not match!" 
                        edge_attr = create_edges_features(extracted_src, extracted_dst, extracted_features, cur_index)
                        
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0) #random node features and 1s for edge_sttr
                    # if snapshot_idx < 5:
                    #     print("edge_attr", edge_attr, edge_attr.shape) # 1, 1..
                    #     print("node_feat", node_feat, node_feat.shape ) #random
                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    if ('edge_attr' not in train_data):
                        # edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)

                        if snapshot_idx-1 ==0:
                            shot_edge_mask = train_edges.t <= train_data['ts_map'][snapshot_idx-1]
                        else:
                            shot_edge_mask = (train_edges.t > train_data['ts_map'][snapshot_idx - 2]) & (train_edges.t <= train_data['ts_map'][snapshot_idx-1])

                        #extracting the right data
                        shot_edge_idx = torch.nonzero(shot_edge_mask, as_tuple=True)[0]
                        shot_edge_times = train_edges.t[shot_edge_idx]
                        extracted_src = train_edges.src[shot_edge_idx]
                        extracted_dst = train_edges.dst[shot_edge_idx]
                        extracted_features = train_edges.msg[shot_edge_idx]
                        
                        edges = torch.stack([extracted_src, extracted_dst], dim=1)
                        # print("edges", edges)
                        
                        unique_edges = torch.unique(edges, dim=0)
                        unique_extracted_src = unique_edges[:, 0]
                        unique_extracted_dst = unique_edges[:, 1]


                        # print("unique_extracted_src", unique_extracted_src, len(unique_extracted_src))
                        # print("unique_extracted_dst", unique_extracted_dst)
                        # print("merged_extracted", merged_extracted)
                        # print("prev_index[0]", prev_index[0])
                        # print("prev_index[1]", prev_index[1])
                        
                        # Compare with prev_index (which should be in the same order)
                        assert torch.equal(unique_extracted_src, prev_index[0][:len(unique_extracted_src)]), "Source nodes do not match!" 
                        assert torch.equal(unique_extracted_dst, prev_index[1][:len(unique_extracted_dst)]), "Target nodes do not match!" 

                        edge_attr = create_edges_features(extracted_src, extracted_dst, extracted_features, prev_index)
                                    
                        
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)
                    # if snapshot_idx < 5:
                    #     print("h", h, h.shape) #torch.Size([352638, 256]) - nodes and features
                    #     print("h_0", h_0, h_0.shape )
                    #     print("c_0", c_0, c_0.shape)
                    #     print("prev_index", prev_index, prev_index.shape) #[][] - (2, edges)
                    # else:
                    if snapshot_idx ==3:
                        break

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)

                neg_dst = torch.randint(
                        0,
                        num_nodes,
                        (pos_index.shape[1],), #num of edges
                        dtype=torch.long,
                        device=args.device,
                    )#neg_dst tensor([190263, 191614, 200024, 197477, 136266, 349326,  80000, 289975],
       # device='cuda:0') torch.Size([8]) ->> nodes indexes?

                e_l_cut = shot_edge_idx + 1
                ts_l_cut = shot_edge_times
                src_l_cut = extracted_src
                tgt_l_cut = extracted_dst
                size = len(src_l_cut)
                _, bad_l_cut = random_sampler_train.sample(size)

                print("size", size, "src_l_cut", src_l_cut)
        
                _, _ = NAT_module.contrast_nat(src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut)
                hop_0 = nat.get_neighborhood_store()[0][0]
                print("hop_0", hop_0)

                pos_out = link_pred(h[pos_index[0]], h[pos_index[1]]) #source nodes to target nodes data from h - probability of future pos_index torch.Size([8, 1])
                neg_out = link_pred(h[pos_index[0]], h[neg_dst])# from target to some random...torch.Size([8, 1])

                loss = criterion(pos_out, torch.ones_like(pos_out))
                loss += criterion(neg_out, torch.zeros_like(neg_out))

                loss.backward()
                optimizer.step()

                total_loss += float(loss) / pos_index.shape[1]


                h_0 = h_0.detach()
                c_0 = c_0.detach()

            train_time = timeit.default_timer() - train_start_time
            print (f'Epoch {epoch}/{num_epochs}, Loss: {total_loss}')
            print ("Train time: ", train_time)
            
            #? Evaluation starts here
            val_snapshots = data['val_data']['edge_index']
            ts_list = data['val_data']['ts_map']
            val_loader = TemporalDataLoader(val_edges, batch_size=batch_size)
            evaluator = Evaluator(name=args.dataset)
            neg_sampler = dataset.negative_sampler
            dataset.load_val_ns()

            start_epoch_val = timeit.default_timer()
            val_metrics, h, h_0, c_0 = test_tgb(h, h_0, c_0, val_loader, val_snapshots, ts_list,
                node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='val')
            val_time = timeit.default_timer() - start_epoch_val
            print(f"Val {metric}: {val_metrics}")
            print ("Val time: ", val_time)
            if (args.wandb):
                wandb.log({"train_loss":(total_loss),
                        "val_" + metric: val_metrics,
                        "train time": train_time,
                        "val time": val_time,
                        })
                
            #! report test results when validation improves
            if (val_metrics > best_val):
                dataset.load_test_ns()
                test_snapshots = data['test_data']['edge_index']
                ts_list = data['test_data']['ts_map']
                test_loader = TemporalDataLoader(test_edges, batch_size=batch_size)
                neg_sampler = dataset.negative_sampler
                dataset.load_test_ns()

                test_start_time = timeit.default_timer()
                test_metrics, h, h_0, c_0 = test_tgb(h, h_0, c_0, test_loader, test_snapshots, ts_list,
                node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='test')
                test_time = timeit.default_timer() - test_start_time
                best_val = val_metrics
                best_test = test_metrics

                print ("test metric is ", test_metrics)
                print ("test elapsed time is ", test_time)
                print ("--------------------------------")
                if ((epoch - best_epoch) >= args.patience and epoch > 1):
                    best_epoch = epoch
                    break
                best_epoch = epoch
        print ("run finishes")
        print ("best epoch is, ", best_epoch)
        print ("best val performance is, ", best_val)
        print ("best test performance is, ", best_test)
        print ("------------------------------------------")
