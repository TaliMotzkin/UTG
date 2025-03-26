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
from typing import List
from timings import *
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'TGX'))
sys.path.insert(0, os.path.join(project_root, 'NAT'))

from torch_geometric.data import TemporalData
writer = SummaryWriter(log_dir="runs/experiment_1")

class IndexedTemporalDataLoader(TemporalDataLoader):
    def __call__(self, arange: List[int]) -> TemporalData:
        start = arange[0]
        end = start + self.events_per_batch
        batch = self.data[start:end]
        
        # This ensures that if a batch starts at index 32, then the indices are [32, 33, ..., 32 + events_per_batch - 1].
        global_edge_indices = torch.arange(start, min(end, len(self.data)), device=batch.src.device)
        batch.edge_indices = global_edge_indices
        
        n_ids = [batch.src, batch.dst]
        if self.neg_sampling_ratio > 0:
            batch.neg_dst = torch.randint(
                low=self.min_dst,
                high=self.max_dst + 1,
                size=(round(self.neg_sampling_ratio * batch.dst.size(0)), ),
                dtype=batch.dst.dtype,
                device=batch.dst.device,
            )
            n_ids += [batch.neg_dst]
        batch.n_id = torch.cat(n_ids, dim=0).unique()
        
        return batch

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
        # print("LSTM<",  x.shape, edge_index.shape, edge_weight.shape)
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

class LinkPredictorWithHop(torch.nn.Module):
    def __init__(self, in_channels, hop_dim, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictorWithHop, self).__init__()
        self.input_dim = in_channels
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.input_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels+1, out_channels))
        self.dropout = dropout
        self.hop_linear1 = torch.nn.Linear(in_channels, hidden_channels)
        self.hop_linear2 = torch.nn.Linear(hidden_channels, 1)


    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, hop):
        # combined_x_i = torch.cat([x_i, source_hop], dim=1)
        # combined_x_j = torch.cat([x_j, target_hop], dim=1)
        
        x = x_i * x_j
        # print("X", x.shape)
        # print("HOP", hop.shape)
        hop = self.hop_linear1(hop)
        hop = F.relu(hop)
        hop = self.hop_linear2(hop)
        # x = torch.cat([x, hop], dim=1)
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # print("X", x.shape)
        # print("hop", hop.shape)
        # x = torch.cat([x, hop.unsqueeze(-1)], dim=1)
        x = torch.cat([x, hop], dim=1)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
        
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



def test_tgb(val_edges,interval, random_sampler, NAT_module, h,
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

    # print("ts_idx", ts_idx)
    # print("ts_list", ts_list)
    k=0
    for batch in test_loader:
        pos_src, pos_dst, pos_t, pos_msg, pos_index = (
        batch.src,
        batch.dst,
        batch.t,
        batch.msg,
        batch.edge_indices 
        )
        #"query_batch" - For each positive edge in the `pos_batch`, return a list of negative edges
       # `split_mode` specifies whether the valiation or test evaluation set should be retrieved.
       # modify now to include edge type argument
        neg_batch_list = neg_sampler.query_batch(np.array(pos_src.cpu()), np.array(pos_dst.cpu()), np.array(pos_t.cpu()), split_mode=split_mode)


        #calculating hop neighboors up to the next ts_idx update
        # if ts_idx == 0:
        #     shot_edge_mask = val_edges.t <= ts_list[ts_idx]
        # else:
        #     shot_edge_mask = (val_edges.t > ts_list[ts_idx - 1]) & (val_edges.t <= ts_list[ts_idx]) 
        # shot_edge_idx = torch.nonzero(shot_edge_mask, as_tuple=True)[0]
        # ts_l_cut  = val_edges.t[shot_edge_idx]
        # src_l_cut  = val_edges.src[shot_edge_idx]
        # tgt_l_cut = val_edges.dst[shot_edge_idx]
        # e_l_cut = shot_edge_idx +1

        
        
        #^^a list of list; each internal list contains the set of negative edges that
                       # should be evaluated against each positive edge.
        for idx, neg_batch in enumerate(neg_batch_list):
            # neg_batch = neg_batch[:995]
            query_src = torch.full((1 + len(neg_batch),), pos_src[idx], device=args.device)
            query_dst = torch.tensor(
                        np.concatenate(
                            ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                            axis=0,
                        ),
                        device=args.device,
                    )
        # size_batch = pos_t.shape[0]
        # neg_batch = neg_batch_list[0]
        # query_src_indices = torch.randint( 0, size_batch, (size_cut,), dtype=torch.long, device=args.device)

            # print("bb", len(neg_batch))
            # print("query_src", query_src, query_src.shape)
            with torch.no_grad():
                NAT_module.eval()
                if getattr(args, "with_hop", 0) == 1: #no need to calculate over the same batch again and again 
                    # size = len(pos_src)
                    # bad_l_cut = neg_batch[:size]
                    # bad_l_cut =torch.tensor(bad_l_cut)
                    size_query = query_src.shape[0]
                    prev_size = size_query
                    time_snap = torch.full((size_query,), pos_t[idx], dtype=torch.long)
                    idx_snap = torch.full((size_query,), pos_index[idx], dtype=torch.long)

                    # print("sizes", query_src[0], query_dst[0], query_dst[0], time_snap[0], idx_snap[0])
                    pos_nat, _ = NAT_module.nat.contrast_nat(query_src, query_dst, query_dst, time_snap, idx_snap, test=True)
                    
                    
                    # cat_true = torch.cat([h[query_src], h[query_dst]], dim=0)
                    # cat_min_true = cat_true.min()
                    # cat_max_true = cat_true.max()
                    # pos_nat = (pos_nat - cat_min_true) / (cat_max_true - cat_min_true)
                    # print("pos_nat", pos_nat.shape, pos_nat, h[query_dst[0]].shape)
                    y_pred =link_pred(h[query_src], h[query_dst], pos_nat) 


                    # cat_fake = torch.cat([h[query_src[1:]], h[query_dst[1:]]], dim=0)
                    # cat_min_fake = cat_fake.min()
                    # cat_max_fake = cat_fake.max()
                    # neg_nat = (neg_nat - cat_min_fake) / (cat_max_fake - cat_min_fake)
                    # y_neg = link_pred(h[query_src[1:]], h[query_dst[1:]], neg_nat)

                    if k <2:
                        print("pos_nat.mean(dim=0, keepdim=True)", pos_nat.mean(dim=0, keepdim=True).mean(), pos_nat.mean(dim=0, keepdim=True).std())
                        # print("neg_nat", neg_nat.mean(dim=0, keepdim=True).mean(), neg_nat.mean(dim=0, keepdim=True).std())
              
                    
                # elif getattr(args, "with_hop", 0) == 1:
                #     # print("query_src.shape[0]", query_src.shape[0])
                #     # if query_src.shape[0] != prev_size:
                #     #     prev_size = query_src.shape[0]
                #     #     # print("making 999", query_src.shape[0], "index", idx )
                #     #     size_query = query_src.shape[0]
                #     #     time_snap = torch.full((size_query,), pos_t[idx], dtype=torch.long)
                #     #     idx_snap = torch.full((size_query,), pos_index[idx], dtype=torch.long)
                #     #     pos_nat, _ = NAT_module.contrast_nat(query_src, query_dst, query_dst, time_snap, idx_snap)
                #         # print("pos_nat", pos_nat.shape)
                #         # print("h[query_src]", h[query_src].shape, h[query_dst].shape)
                #     y_pos = link_pred(h[query_src[0]], h[query_dst[0]], pos_nat)
                #     y_neg = link_pred(h[query_src[1:]], h[query_dst[1:]], bad_nat)
                    # y_pred =link_pred(h[query_src], h[query_dst], pos_nat) 
                    
                else:
                    # y_pos = link_pred(h[query_src[0]], h[query_dst[0]])
                    # y_neg = link_pred(h[query_src[1:]], h[query_dst[1:]])
                    y_pred =link_pred(h[query_src], h[query_dst]) 
                    
            # y_pos = y_pos.squeeze(dim=-1).detach()
            # y_neg = y_neg.squeeze(dim=-1).detach()
            y_pred = y_pred.squeeze(dim=-1).detach()

            if k < 2:
                print("y_pred", y_pred[0].mean(dim=0),idx)
                print("y_neg", y_pred[1:].mean(dim=0))

            input_dict = {
            "y_pred_pos": np.array( y_pred[0].cpu()),
            "y_pred_neg": np.array( y_pred[1:].cpu()),
            "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
            k +=1

        #* update the model now if the prediction batch has moved to next snapshot
        while (pos_t[-1] > ts_list[ts_idx] and ts_idx < max_ts_idx):
            # print("pos_t[-1]", pos_t[-1], ts_list[ts_idx])
            with torch.no_grad():
                # print("val_edges", val_edges[0:200].src, val_edges[0:200].dst, val_edges[0:200].t)
                cur_index = test_snapshots[ts_idx]
                cur_index = cur_index.long().to(args.device)
                # edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
                
                # if ts_idx == 0:
                #     shot_edge_mask = val_edges.t < ts_list[ts_idx]
                # elif ts_idx == min(list(ts_list.keys())):
                #     minimal_time = max(0, min( ts_list[ts_idx],  ts_list[ts_idx] - interval))
                #     shot_edge_mask = (val_edges.t >= minimal_time) & (val_edges.t <= ts_list[ts_idx])
                # else:
                #     shot_edge_mask = (val_edges.t >= ts_list[ts_idx - 1]) & (val_edges.t <= ts_list[ts_idx])
                # shot_edge_idx = torch.nonzero(shot_edge_mask, as_tuple=True)[0]
                # extracted_src = val_edges.src[shot_edge_idx] 
                # extracted_dst = val_edges.dst[shot_edge_idx] 
                # extracted_features = val_edges.msg[shot_edge_idx]

                # print("extracted_src", extracted_src, extracted_src.shape)
                # print("extracted_dst", extracted_dst, extracted_dst.shape)
                # print("extracted_features", extracted_features, extracted_features.shape)
                # print("cur_index", cur_index, cur_index.shape)
                edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
                # edge_attr = create_edges_features(extracted_src, extracted_dst, extracted_features, cur_index)

                # print("node_feat", node_feat.shape)
                # print("cur_index", cur_index, cur_index.shape)
                # print("edge_attr", edge_attr.shape)
                h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0)
                h = h.detach()
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            ts_idx += 1
            # print("ts_idx", ts_idx)

    #* update to the final snapshot
    with torch.no_grad():
        # print("max_ts_idx", max_ts_idx)
        cur_index = test_snapshots[max_ts_idx]
        cur_index = cur_index.long().to(args.device)
        # shot_edge_mask = (val_edges.t >= ts_list[max_ts_idx - 1]) & (val_edges.t <= ts_list[max_ts_idx])
        # shot_edge_idx = torch.nonzero(shot_edge_mask, as_tuple=True)[0]
        # extracted_src = val_edges.src[shot_edge_idx]
        # extracted_dst = val_edges.dst[shot_edge_idx]
        # extracted_features = val_edges.msg[shot_edge_idx]
        # edge_attr = create_edges_features(extracted_src, extracted_dst, extracted_features, cur_index)

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
    # print("num_unique_edges", num_unique_edges)
    # print("unique_edges", unique_edges, unique_edges.shape)
    # print("inverse_indices", inverse_indices, inverse_indices.shape)
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
            print(src_val, dst_val)
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

    # print("print data properties in the 0 link", full_data.src [0],full_data.dst[0],full_data.t[0],
    #             full_data.msg[0],
    #             full_data.y[0])

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
    num_nodes = data['train_data']['num_nodes'] 
        #NOt sure why in UTG they are adding 1?..
    # num_nodes = data['train_data']['num_nodes'] + 1

    interval = time_intervals(args.time_scale)

    #this is not working since in the negative batch (query_batch) sampling they are using the past framework)
    # train_ts_map = train_data['ts_map']  # e.g., {0: 0, 1: 3600, 2: 7200, ...}
    # train_min_time = max(0,min(min(train_ts_map.values()),min(train_ts_map.values())- interval))
    # train_max_time = max(train_ts_map.values())
    # train_mask = (full_data.t >= train_min_time) & (full_data.t < train_max_time)

    # val_ts_map = val_data['ts_map']
    # val_min_time = max(0,min(min(val_ts_map.values()),min(val_ts_map.values())- interval))
    # val_max_time = max(val_ts_map.values())
    # val_mask = (full_data.t >= val_min_time) & (full_data.t < val_max_time)

    # test_ts_map = test_data['ts_map']
    # test_min_time = max(0,min(min(test_ts_map.values()),min(test_ts_map.values())- interval))
    # test_max_time = max(test_ts_map.values())
    # test_mask = (full_data.t >= test_min_time) & (full_data.t <= test_max_time)

    # train_edges = full_data[train_mask]
    # val_edges = full_data[val_mask]
    # test_edges = full_data[test_mask]

    # print("train_edges!!", train_edges)
    # print("val_edges", val_edges.t)
    e_feat = full_data.msg
    edge_feat_dim = e_feat.shape[1]
    if args.pre_training:
        NAT_module = init_nat_module(full_data, e_feat, node_feat, train_edges,val_edges, test_edges, args, argsv)
        NAT_module.logger.info(f"learning_rate {args.lr} \n, architecture gclstm \n dataset {args.dataset} \n time granularity  {args.time_scale}\n")
        modelN = NAT_module.nat
        modelN.load_state_dict(torch.load("./nat_models/model_state_dict.pth"))
        modelN.eval()
    else:     
        NAT_module = init_nat_module(full_data, e_feat, node_feat, train_edges,val_edges, test_edges, args, argsv)
        NAT_module.logger.info(f"learning_rate {args.lr} \n, architecture gclstm \n dataset {args.dataset} \n time granularity  {args.time_scale}\n")


    random_sampler_train = RandEdgeSampler((train_edges.src.cpu().numpy(), ), (train_edges.dst.cpu().numpy(), ))
    random_sampler_test = RandEdgeSampler((test_edges.src.cpu().numpy(), ), (test_edges.dst.cpu().numpy(), ))
    random_sampler_val = RandEdgeSampler((val_edges.src.cpu().numpy(), ), (val_edges.dst.cpu().numpy(), ))
    num_epochs = args.max_epoch
    lr = args.lr

    runs_best_val = []
    runs_best_test = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print (f"Run {seed}")
        
        #* initialization of the model to prep for training
        if args.with_hop == 0:
            model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=1).to(args.device)
        else:
            model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=node_feat_dim+2*args.self_dim, K=1).to(args.device)
            # model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=node_feat_dim+args.ngh_dim+args.pos_dim, K=1).to(args.device)
        node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)

        if args.with_hop == 0:
            link_pred = LinkPredictor(hidden_dim, hidden_dim, 1,
                                2, 0.2).to(args.device)
        if args.with_hop == 1:
            link_pred = LinkPredictorWithHop(in_channels=node_feat_dim+2*args.self_dim, 
                                 hop_dim=7, 
                                 hidden_channels=hidden_dim, 
                                 out_channels=1, 
                                 num_layers=2, 
                                 dropout=0.2).to(args.device)
            # link_pred = LinkPredictorWithHop(in_channels=2*(node_feat_dim+args.ngh_dim+args.pos_dim), 
            #                      hop_dim=7, 
            #                      hidden_channels=hidden_dim, 
            #                      out_channels=1, 
            #                      num_layers=2, 
            #                      dropout=0.2).to(args.device)


        optimizer = torch.optim.Adam(
            set(model.parameters()) | set(link_pred.parameters()), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val = 0
        best_test = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            print ("------------------------------------------")
            # NAT_module.nat.set_seed(seed)

            # NAT_module.nat.reset_store()
            # NAT_module.nat.reset_self_rep()
            # NAT_module.nat.train()
            train_start_time = timeit.default_timer()
            optimizer.zero_grad()
            total_loss = 0
            model.train()
            link_pred.train()
            timestep_list = train_data["ts_map"]
            snapshot_list = train_data['edge_index'] #0: snap, 1: snap.....207:snap
            # print("time", train_data['ts_map']) #{0: 0, 1: 3600, 2: 7200, 3: 10800...207:1861200}
            
            h_0, c_0, h = None, None, None
            total_loss = 0
            k= 0
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

                        # edges = torch.stack([extracted_src, extracted_dst], dim=1)
                        # # print("edges", edges)
                        
                        # unique_edges = torch.unique(edges, dim=0)
                       
                        # unique_extracted_src = unique_edges[:, 0]
                        # unique_extracted_dst = unique_edges[:, 1]

                        # merged_extracted = torch.cat([unique_extracted_src, unique_extracted_dst])
                        # # Compare with cur_index (which should be in the same order)
                        # assert torch.equal(unique_extracted_src, cur_index[0][:len(unique_extracted_src)]), "Source nodes do not match!" 
                        # assert torch.equal(unique_extracted_dst, cur_index[1][:len(unique_extracted_dst)]), "Target nodes do not match!" 
                        # print("train_edges", train_edges[450:].src, train_edges[450:].dst, val_edges[450:].t)
                        # print("train_data", train_data["ts_map"])
                        # edge_attr = create_edges_features(extracted_src, extracted_dst, extracted_features, cur_index)
                        
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
                        edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)

                        # if snapshot_idx-1 ==0:
                        #     shot_edge_mask = train_edges.t <= train_data['ts_map'][snapshot_idx-1]
                        # else:
                        shot_edge_mask = (train_edges.t > train_data['ts_map'][snapshot_idx - 1]) & (train_edges.t <= train_data['ts_map'][snapshot_idx])

                        #extracting the right data
                        shot_edge_idx = torch.nonzero(shot_edge_mask, as_tuple=True)[0]
                        shot_edge_times = train_edges.t[shot_edge_idx]
                        extracted_src = train_edges.src[shot_edge_idx]
                        extracted_dst = train_edges.dst[shot_edge_idx]
                        extracted_features = train_edges.msg[shot_edge_idx]
                        
                        # edges = torch.stack([extracted_src, extracted_dst], dim=1)
                        # # print("edges", edges)
                        
                        # unique_edges = torch.unique(edges, dim=0)
                        # unique_extracted_src = unique_edges[:, 0]
                        # unique_extracted_dst = unique_edges[:, 1]


                        # # print("unique_extracted_src", unique_extracted_src, len(unique_extracted_src))
                        # # print("unique_extracted_dst", unique_extracted_dst)
                        # # print("merged_extracted", merged_extracted)
                        # # print("prev_index[0]", prev_index[0])
                        # # print("prev_index[1]", prev_index[1])
                        
                        # # Compare with prev_index (which should be in the same order)
                        # assert torch.equal(unique_extracted_src, prev_index[0][:len(unique_extracted_src)]), "Source nodes do not match!" 
                        # assert torch.equal(unique_extracted_dst, prev_index[1][:len(unique_extracted_dst)]), "Target nodes do not match!" 

                        # edge_attr = create_edges_features(extracted_src, extracted_dst, extracted_features, prev_index)
                                    
                        
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)
                    # if snapshot_idx < 5:
                    #     print("h", h, h.shape) #torch.Size([352638, 256]) - nodes and features
                    #     print("h_0", h_0, h_0.shape )
                    #     print("c_0", c_0, c_0.shape)
                    #     print("prev_index", prev_index, prev_index.shape) #[][] - (2, edges)
                    # else:
                    # if snapshot_idx ==3:
                    #     break

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)               
                
                

                # print("size", size, "src_l_cut", src_l_cut)
                
                if args.with_hop ==0:
                    neg_dst = torch.randint( 0, num_nodes, (pos_index.shape[1],), dtype=torch.long, device=args.device)
                    pos_out = link_pred(h[pos_index[0]], h[pos_index[1]]) #source nodes to target nodes data from h - probability of future pos_index torch.Size([8, 1])
                    neg_out = link_pred(h[pos_index[0]], h[neg_dst])# from target to some random...torch.Size([8, 1])
                if args.with_hop ==1:
                    e_l_cut = shot_edge_idx + 1
                    ts_l_cut = shot_edge_times
                    src_l_cut = extracted_src
                    tgt_l_cut = extracted_dst
                    size_snap = pos_index.shape[1]
            
                    size_cut = len(src_l_cut)
                    # if size_cut > size_snap:
                    neg_dst = torch.randint( 0, num_nodes, (size_snap,), dtype=torch.long, device=args.device)
                    indices = torch.randint( 0, size_snap, (size_cut,), dtype=torch.long, device=args.device)
                    bad_l_cut = neg_dst[indices]
                    
                    # else:
                    #     neg_dst = torch.randint( 0, num_nodes, (size_snap,), dtype=torch.long, device=args.device)
                    #     bad_l_cut = neg_dst[:size_cut]
                    time_snap = torch.full((size_snap,), timestep_list[snapshot_idx], dtype=torch.long)
                    
                    edge_snap = torch.arange(snapshot_idx, snapshot_idx + size_snap, dtype=torch.long)
                    pos_hop , neg_hop = NAT_module.nat.contrast_nat(pos_index[0], pos_index[1], neg_dst, time_snap, edge_snap, test =True)
                    # print("H", h.shape)
                    # print("neg_dst", neg_dst.shape, edge_snap.shape, time_snap.shape)
                    # print("pos_hop", pos_hop.shape)
                    # print("neg_hop", neg_hop.shape)
                    # print("h[pos_index[0]]", h[pos_index[0]].shape)
                    # print("h[pos_index[0]]", h[neg_dst].shape)

                    # cat_true = torch.cat([h[pos_index[0]], h[pos_index[1]]], dim=0)
                    # cat_min_true = cat_true.min()
                    # cat_max_true = cat_true.max()
                    # pos_hop = (pos_hop - cat_min_true) / (cat_max_true - cat_min_true)

                    # cat_fake = torch.cat([h[pos_index[0]], h[neg_dst]], dim=0)
                    # cat_min_fake = cat_fake.min()
                    # cat_max_fake = cat_fake.max()
                    # neg_hop = (neg_hop - cat_min_fake) / (cat_max_fake - cat_min_fake)

                    pos_out = link_pred(h[pos_index[0]], h[pos_index[1]],pos_hop)
                    neg_out = link_pred(h[pos_index[0]], h[neg_dst], neg_hop)

                    if k <2:
                        print("pos_hop", pos_hop.mean(dim=0, keepdim=True).mean(), pos_hop.mean(dim=0, keepdim=True).std())
                        print("neg_hop", neg_hop.mean(dim=0, keepdim=True).mean(), neg_hop.mean(dim=0, keepdim=True).std())
                # print("h[pos_index[0]]", h[pos_index[0]])
                # print(" h[pos_index[1]]", h[pos_index[1]])

                    # print("pos_out", pos_out.shape, pos_out.mean(dim=0), snapshot_idx)
                    # print("neg_out", neg_out.shape, neg_out.mean(dim=0))
                loss = criterion(pos_out, torch.ones_like(pos_out))
                loss += criterion(neg_out, torch.zeros_like(neg_out))

                loss.backward()
                optimizer.step()

                total_loss += float(loss) / pos_index.shape[1]


                h_0 = h_0.detach()
                c_0 = c_0.detach()
                k +=1

            train_time = timeit.default_timer() - train_start_time
            print (f'Epoch {epoch}/{num_epochs}, Loss: {total_loss}')
            print ("Train time: ", train_time)


            #? Evaluation starts here
    #         val_snapshots = data['val_data']['edge_index']
    #         ts_list = data['val_data']['ts_map']
    #         val_loader = IndexedTemporalDataLoader(val_edges, batch_size=batch_size)
    #         evaluator = Evaluator(name=args.dataset)
    #         neg_sampler = dataset.negative_sampler
    #         dataset.load_val_ns()

    #         start_epoch_val = timeit.default_timer()
    #         val_metrics, h, h_0, c_0 = test_tgb(full_data, interval,random_sampler_val, NAT_module, h, h_0, c_0, val_loader, val_snapshots, ts_list,
    #             node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='val')
    #         val_time = timeit.default_timer() - start_epoch_val
    #         print(f"Val {metric}: {val_metrics}")
    #         print ("Val time: ", val_time)
    #         NAT_module.logger.info(f"train_loss: {total_loss} \n, metric: {val_metrics}\n  train time: {train_time} \n val time: {val_time} \n ")
    #         if (args.wandb):
    #             wandb.log({"train_loss":(total_loss),
    #                     "val_" + metric: val_metrics,
    #                     "train time": train_time,
    #                      "val time": val_time,
    #                     })
    #         writer.add_scalar("Loss/train", total_loss, epoch)
    #         writer.add_scalar("MRR/val", val_metrics, epoch)
    #         #! report test results when validation improves
    #         if (val_metrics > best_val):
    #             dataset.load_test_ns()
    #             test_snapshots = data['test_data']['edge_index']
    #             ts_list = data['test_data']['ts_map']
    #             test_loader = IndexedTemporalDataLoader(test_edges, batch_size=batch_size)
    #             neg_sampler = dataset.negative_sampler
    #             dataset.load_test_ns()

    #             test_start_time = timeit.default_timer()
    #             test_metrics, h, h_0, c_0 = test_tgb(full_data, interval,random_sampler_test, NAT_module,h, h_0, c_0, test_loader, test_snapshots, ts_list,
    #             node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='test')
    #             test_time = timeit.default_timer() - test_start_time
    #             best_val = val_metrics
    #             best_test = test_metrics

    #             writer.add_scalar("MRR/test", test_metrics, epoch)

    #             print ("test metric is ", test_metrics)
    #             print ("test elapsed time is ", test_time)
    #             print ("--------------------------------")
    #             if ((epoch - best_epoch) >= args.patience and epoch > 1):
    #                 best_epoch = epoch
    #                 break
    #             best_epoch = epoch
    #             NAT_module.logger.info(f"best epoch {best_epoch} \n, test_metrics {test_metrics} \n, test_time {test_time}\n")
    #             if (args.wandb):
    #                 wandb.log({"best epoch":(best_epoch),
    #                         "test_metrics": test_metrics,
    #                          "test_time": test_time,
    #                         })
    #     print ("run finishes")
    #     print ("best epoch is, ", best_epoch)
    #     print ("best val performance is, ", best_val)
    #     print ("best test performance is, ", best_test)
    #     print ("------------------------------------------")

    #     runs_best_val.append(best_val)
    #     runs_best_test.append(best_test)
    # runs_best_val = np.array(runs_best_val)
    # runs_best_test = np.array(runs_best_test)
    
    # val_mean, val_std = runs_best_val.mean(), runs_best_val.std(ddof=1)
    # test_mean, test_std = runs_best_test.mean(), runs_best_test.std(ddof=1)
    
    # print("========================================")
    # print(f"Summary of {args.num_runs} runs:")
    # print(f"Val MRR: mean={val_mean:.4f} ± {val_std:.4f}")
    # print(f"Test MRR: mean={test_mean:.4f} ± {test_std:.4f}")
    # print("========================================")
    
    # writer.close()
    
################################# non val @############################
            #? Evaluation starts here
            # val_snapshots = data['val_data']['edge_index']
            # ts_list = data['val_data']['ts_map']
            # val_loader = IndexedTemporalDataLoader(val_edges, batch_size=batch_size)
            evaluator = Evaluator(name=args.dataset)
            neg_sampler = dataset.negative_sampler
            # dataset.load_val_ns()

            # start_epoch_val = timeit.default_timer()
            # val_metrics, h, h_0, c_0 = test_tgb(full_data, interval,random_sampler_val, NAT_module, h, h_0, c_0, val_loader, val_snapshots, ts_list,
                # node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='val')
            # val_time = timeit.default_timer() - start_epoch_val
            # print(f"Val {metric}: {val_metrics}")
            # print ("Val time: ", val_time)
            if (args.wandb):
                wandb.log({"train_loss":(total_loss),
                        "val_" + metric: val_metrics,
                        "train time": train_time,
                        # "val time": val_time,
                        })
            writer.add_scalar("Loss/train", total_loss, epoch)
            # writer.addReport_scalar("MRR/val", val_metrics, epoch)
            #! report test results when validation improves
            # if (val_metrics > best_val):
            if epoch % 50 == 0 :
                dataset.load_test_ns()
                test_snapshots = data['test_data']['edge_index']
                ts_list = data['test_data']['ts_map']
                test_loader = IndexedTemporalDataLoader(test_edges, batch_size=batch_size)
                neg_sampler = dataset.negative_sampler
                dataset.load_test_ns()

                test_start_time = timeit.default_timer()
                test_metrics, h, h_0, c_0 = test_tgb(full_data, interval,random_sampler_test, modelN,h, h_0, c_0, test_loader, test_snapshots, ts_list,
                node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='test')
                test_time = timeit.default_timer() - test_start_time
                # best_val = val_metrics
                best_test = test_metrics

                writer.add_scalar("MRR/test", test_metrics, epoch)

                print ("test metric is ", test_metrics)
                print ("test elapsed time is ", test_time)
                print ("--------------------------------")
                # if ((epoch - best_epoch) >= args.patience and epoch > 1):
                #     best_epoch = epoch
                #     break
                best_epoch = epoch
        print ("run finishes")
        print ("best epoch is, ", best_epoch)
        # print ("best val performance is, ", best_val)
        print ("best test performance is, ", best_test)
        print ("------------------------------------------")

        # runs_best_val.append(best_val)
        runs_best_test.append(best_test)
    # runs_best_val = np.array(runs_best_val)
    runs_best_test = np.array(runs_best_test)
    
    # val_mean, val_std = runs_best_val.mean(), runs_best_val.std(ddof=1)
    test_mean, test_std = runs_best_test.mean(), runs_best_test.std(ddof=1)
    
    print("========================================")
    print(f"Summary of {args.num_runs} runs:")
    # print(f"Val MRR: mean={val_mean:.4f} ± {val_std:.4f}")
    print(f"Test MRR: mean={test_mean:.4f} ± {test_std:.4f}")
    print("========================================")
    
    writer.close()
