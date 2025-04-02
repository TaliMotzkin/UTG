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
        
class PositionalEncoding(nn.Module):
      def __init__(self, d_model, dropout= 0.1, max_length= 5000):
        super().__init__()     
           
        self.dropout = nn.Dropout(p=dropout)      
        pe = torch.zeros(max_length, d_model)    
        k = torch.arange(0, max_length).unsqueeze(1)  
        div_term = torch.exp(                                 
                torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(k * div_term)    
        pe[:, 1::2] = torch.cos(k * div_term)  
    
        pe = pe.unsqueeze(0)          
        self.register_buffer("pe", pe)                        
    
      def forward(self, x):
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
        return self.dropout(x)
      
class LinkPredictorWithHop(torch.nn.Module):
    def __init__(self, mode, in_channels, hop_dim, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictorWithHop, self).__init__()
        self.input_dim = in_channels
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.input_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        
        self.dropout = dropout

        self.mode = mode
        

        if self.mode == "exp_1":
            self.hop_linear1 = torch.nn.Linear(hop_dim, 16)
            self.hop_linear2 = torch.nn.Linear(16, 8)
            self.lins.append(torch.nn.Linear(hidden_channels+8, out_channels))

        elif self.mode == "pos":
            self.pos_32 = PositionalEncoding(d_model=1)
            self.hop_linear2 = torch.nn.Linear(hop_dim, hidden_channels)
            self.lins.append(torch.nn.Linear(hidden_channels+hidden_channels, out_channels))

            
        else:
            self.hop_linear1 = torch.nn.Linear(hop_dim, hidden_channels)
            self.hop_linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
            self.lins.append(torch.nn.Linear(hidden_channels+hidden_channels, out_channels))

        
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, hop):
        # combined_x_i = torch.cat([x_i, source_hop], dim=1)
        # combined_x_j = torch.cat([x_j, target_hop], dim=1)
        
        x = x_i * x_j
        # print("X", x.shape)
        # print("HOP", hop.shape)
        if self.mode == "pos":
            hop = self.pos_32(hop.unsqueeze(-1)).squeeze(-1)
            # print("hop", hop.shape)
        else:
            hop = self.hop_linear1(hop)
            # print("hop", hop.shape)
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



def test_tgb(exp_name, val_edges,interval, random_sampler, h,
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
             NAT_module = None,
             split_mode='val'):
    
    model.eval()
    link_pred.eval()

    perf_list = []
    ts_idx = min(list(ts_list.keys()))
    max_ts_idx = max(list(ts_list.keys()))

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
                
                if getattr(args, "with_hop", 0) == 1: 
                    NAT_module.eval()
                    size_query = query_src.shape[0]
                    # print("size_query", size_query)
                    prev_size = size_query
                    time_snap = torch.full((size_query,), pos_t[idx], dtype=torch.long)
                    idx_snap = torch.full((size_query,), pos_index[idx], dtype=torch.long)

                    if args.exp_name =="exp_2":
                        _, _, joint_p, ngh_and_batch_id_p = NAT_module.contrast(exp_name, args.pre_training, query_src[0].unsqueeze(0), query_dst[0].unsqueeze(0), query_dst[0].unsqueeze(0), time_snap[0].unsqueeze(0), idx_snap[0].unsqueeze(0), test=True)
                        joint_n, ngh_and_batch_id_n, _, _ = NAT_module.contrast(exp_name, args.pre_training, query_src[1:], query_dst[1:], query_dst[1:], time_snap[1:], idx_snap[1:], test=True)
                        _, ngh_and_batch_id_p = torch.unique(ngh_and_batch_id_p[:,1], return_inverse=True)
                        _, ngh_and_batch_id_n = torch.unique(ngh_and_batch_id_n[:,1], return_inverse=True)

                        pos_nat = scatter(joint_p, ngh_and_batch_id_p, dim=0, reduce="mean")
                        neg_nat = scatter(joint_n, ngh_and_batch_id_n, dim=0, reduce="mean")

                    elif args.exp_name =="exp_1":
                        modelN.set_seed(1)
                        pos_nat = None
                        neg_nat = None
                        for hop in range(args.n_hop):
                            hop = int(hop)
                            # print("hping", pos_index[0] , args.n_degree[hop] , modelN.ncache_hash(pos_index[1], hop+1))
                            idx_pos = query_src[0].unsqueeze(0) * int(args.n_degree[hop]) + modelN.ncache_hash(query_dst[0].unsqueeze(0), hop+1)
                            idx_neg = query_src[1:] * int(args.n_degree[hop]) + modelN.ncache_hash(query_dst[1:], hop+1)
                            vec_pos= modelN.get_neighborhood_store()[hop+1][idx_pos,3:]
                            vec_neg = modelN.get_neighborhood_store()[hop+1][idx_neg,3:]
                            if pos_nat is None:
                         
                                pos_nat = vec_pos.clone()
                                neg_nat = vec_neg.clone()
                            else:
                                pos_nat = torch.cat([pos_nat, vec_pos], dim=-1)  # concat along last dim
                                neg_nat = torch.cat([neg_nat, vec_neg], dim=-1)
                                                
                    else:
                        pos_nat, _ = NAT_module.contrast(exp_name, args.pre_training, query_src[0].unsqueeze(0), query_dst[0].unsqueeze(0), query_dst[0].unsqueeze(0), time_snap[0].unsqueeze(0), idx_snap[0].unsqueeze(0), test=True)
                        _, neg_nat = NAT_module.contrast(exp_name, args.pre_training, query_src[1:], query_dst[1:], query_dst[1:], time_snap[1:], idx_snap[1:], test=True)
                    
                    cat_min_true = pos_nat.min()
                    cat_max_true = pos_nat.max()
                    pos_nat = (pos_nat - cat_min_true) / (cat_max_true - cat_min_true)
                    y_pos =link_pred(h[query_src[0]].unsqueeze(0), h[query_dst[0]].unsqueeze(0), pos_nat) 

                    cat_min_fake = neg_nat.min()
                    cat_max_fake = neg_nat.max()
                    neg_nat = (neg_nat - cat_min_fake) / (cat_max_fake - cat_min_fake)
                    y_neg = link_pred(h[query_src[1:]], h[query_dst[1:]], neg_nat)

                    if k <2:
                        print("pos_nat.mean(dim=0, keepdim=True)", pos_nat.mean(dim=0, keepdim=True).mean(), pos_nat.mean(dim=0, keepdim=True).std())
                        print("neg_nat", neg_nat.mean(dim=0, keepdim=True).mean(), neg_nat.mean(dim=0, keepdim=True).std())
                    
                else:
                    y_pos = link_pred(h[query_src[0]], h[query_dst[0]])
                    y_neg = link_pred(h[query_src[1:]], h[query_dst[1:]])
                    
            y_pos = y_pos.squeeze(dim=-1).detach()
            y_neg = y_neg.squeeze(dim=-1).detach()

            input_dict = {
            "y_pred_pos": np.array(y_pos.cpu()),
            "y_pred_neg": np.array(y_neg.cpu()),
            "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
            k +=1

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
    import datetime
    import sys
    from torch_scatter import scatter



    args, argsv = get_args()
    set_random(args.seed)

    batch_size = args.batch_size

    dataset = PyGLinkPropPredDataset(name=args.dataset, root="datasets") #time step,source node,target node,weight of nodes as seen in the csv
    full_data = dataset.get_TemporalData() 
    full_data = full_data.to(args.device) #TemporalData(src=[4873540], dst=[4873540], t=[4873540], msg=[4873540, 1], y=[4873540]) --> no w, no type!! to get data aboput those attributes just  .src[index]

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
    
    hidden_dim = 64
    
    #* load the discretized version
    data = loader(dataset=args.dataset, time_scale=args.time_scale) #loaded adges - 4873540

    train_data = data['train_data'] 
    val_data = data['val_data'] 
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] 
  

    interval = time_intervals(args.time_scale)
    
    e_feat = full_data.msg
    edge_feat_dim = 1

    n_str = ""
    for degree in args.n_degree:
        n_str += (str(degree) + "k")

    if args.with_hop ==1:        
        if args.pre_training == "use_pretrain":
            NAT_module = init_nat_module(full_data, e_feat, node_feat, train_edges,val_edges, test_edges, args, argsv)
            NAT_module.logger.info(f"learning_rate {args.lr} \n, architecture gclstm \n dataset {args.dataset} \n time granularity  {args.time_scale} {args.pre_training}\n")
            modelN = NAT_module.nat
            modelN.load_state_dict(torch.load(f"./nat_models/pre_trained-{n_str}-{args.dataset}.pth"))
            print("model uploaded", f"pre_trained-{n_str}-{args.dataset}.pth")
            modelN.eval()
    
        if args.pre_training == "dont_use":     
            NAT_module = init_nat_module(full_data, e_feat, node_feat, train_edges,val_edges, test_edges, args, argsv)
            NAT_module.logger.info(f"learning_rate {args.lr} \n, architecture gclstm \n dataset {args.dataset} \n time granularity  {args.time_scale} {args.pre_training}\n")
            modelN = NAT_module.nat
            
        if args.pre_training == "train":
            NAT_module = init_nat_module(full_data, e_feat, node_feat, train_edges,val_edges, test_edges, args, argsv)
            NAT_module.logger.info(f"learning_rate {args.lr} \n, architecture gclstm \n dataset {args.dataset} \n time granularity  {args.time_scale} {args.pre_training}\n")
    
            train_size  = train_edges.src.shape[0]
            train_data = train_edges.src.cpu().numpy() , train_edges.dst.cpu().numpy(), train_edges.t.cpu().numpy(), torch.arange(0, train_size).cpu().numpy() ,train_edges.y.cpu().numpy()
            val_size = val_edges.src.shape[0]
            val_data = val_edges.src.cpu().numpy() , val_edges.dst.cpu().numpy(), val_edges.t.cpu().numpy(), torch.arange(0, val_size).cpu().numpy() ,val_edges.y.cpu().numpy()
            train_val_data = (train_data, val_data)
    
            random_sampler_train = RandEdgeSampler((train_edges.src.cpu().numpy(), ), (train_edges.dst.cpu().numpy(), ))
            random_sampler_val = RandEdgeSampler((val_edges.src.cpu().numpy(), ), (val_edges.dst.cpu().numpy(), ))
            rand_samplers = random_sampler_train, random_sampler_val
            optimizer = torch.optim.Adam(NAT_module.nat.parameters(), lr=args.lr)
            criterion = torch.nn.BCELoss()
            early_stopper = EarlyStopMonitor(tolerance=1e-3)
            logger = NAT_module.logger
            NUM_HOP = args.n_hop
            NUM_EPOCH = args.max_epoch
            BATCH_SIZE = args.batch_size
            
            train_val(args.exp_name, args.pre_training, train_val_data, NAT_module.nat, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, rand_samplers, logger, 0, n_hop=NUM_HOP)
            
            logger.info('Saving NAT model ...')
            torch.save(NAT_module.nat.state_dict(), f"./nat_models/pre_trained-{n_str}-{args.dataset}.pth")
            logger.info('NAT model saved')
            sys.exit()
            
        
    random_sampler_train = RandEdgeSampler((train_edges.src.cpu().numpy(), ), (train_edges.dst.cpu().numpy(), ))
    random_sampler_test = RandEdgeSampler((test_edges.src.cpu().numpy(), ), (test_edges.dst.cpu().numpy(), ))
    random_sampler_val = RandEdgeSampler((val_edges.src.cpu().numpy(), ), (val_edges.dst.cpu().numpy(), ))
    
    num_epochs = args.max_epoch
    lr = args.lr

    runs_best_val = []
    runs_best_test = []
    runs_best_times = []
    base_dir = f"runs/{datetime.datetime.now().strftime('%m%d-%H%M')}-"
    print("base dir", base_dir)
    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)        
        
        current_exp_dir = f"{args.with_hop}-{seed}-{n_str}-{args.lr}-{args.exp_name}-{args.dataset}-{args.time_scale}"
        save_path = base_dir + "/" + current_exp_dir + "/"
        writer = SummaryWriter(log_dir=f"runs/{save_path}")

        print (f"Run {seed}")
        
        #* initialization of the model to prep for training
        if args.with_hop == 0:
            model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=1).to(args.device)
            link_pred = LinkPredictor(hidden_dim, hidden_dim, 1,
                                2, 0.2).to(args.device)
        if args.with_hop == 1:
            
            if args.exp_name == "exp_2":
                model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=1).to(args.device)
                link_pred = LinkPredictorWithHop(args.pos, in_channels=hidden_dim, 
                                 hop_dim=node_feat_dim+args.pos_dim+4, 
                                 hidden_channels=hidden_dim, 
                                 out_channels=1, 
                                 num_layers=2, 
                                 dropout=0.2).to(args.device)
              
            elif args.exp_name == "exp_1":
                model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=node_feat_dim+args.pos_dim+4, K=1).to(args.device)
                link_pred = LinkPredictorWithHop("exp_1", in_channels=node_feat_dim+args.pos_dim+4, 
                                     hop_dim=4*args.n_hop, 
                                     hidden_channels=hidden_dim, 
                                     out_channels=1, 
                                     num_layers=2, 
                                     dropout=0.2).to(args.device)
            else:
                model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=1).to(args.device) 
                link_pred =LinkPredictorWithHop("not_pos",in_channels=hidden_dim, 
                                 hop_dim=node_feat_dim+2*args.self_dim, 
                                 hidden_channels=hidden_dim, 
                                 out_channels=1, 
                                 num_layers=2, 
                                 dropout=0.2).to(args.device)
                
                
        node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)

        optimizer = torch.optim.Adam(
            set(model.parameters()) | set(link_pred.parameters()), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val = 0
        best_test = 0
        best_epoch = 0
        best_test_time = 0

        for epoch in range(num_epochs):
            print ("------------------------------------------")
            if args.pre_training == "dont_use" and args.with_hop ==1:
                modelN.set_seed(seed)
                modelN.reset_store()
                modelN.reset_self_rep()
                modelN.train()
            train_start_time = timeit.default_timer()
            optimizer.zero_grad()
            total_loss = 0
            model.train()
            link_pred.train()
            timestep_list = train_data["ts_map"]
            snapshot_list = train_data['edge_index'] #0: snap, 1: snap.....207:snap
            
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
                        
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0) #random node features and 1s for edge_sttr
                
                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    if ('edge_attr' not in train_data):
                        edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)

                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)
        

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)               
                
                if args.with_hop ==0:
                    neg_dst = torch.randint( 0, num_nodes, (pos_index.shape[1],), dtype=torch.long, device=args.device)
                    pos_out = link_pred(h[pos_index[0]], h[pos_index[1]]) #source nodes to target nodes data from h - probability of future pos_index torch.Size([8, 1])
                    neg_out = link_pred(h[pos_index[0]], h[neg_dst])# from target to some random...torch.Size([8, 1])
                if args.with_hop ==1:
                    
                    size_snap = pos_index.shape[1]
                    neg_dst = torch.randint( 0, num_nodes, (size_snap,), dtype=torch.long, device=args.device)

                    time_snap = torch.full((size_snap,), timestep_list[snapshot_idx], dtype=torch.long)
                    edge_snap = torch.arange(snapshot_idx, snapshot_idx + size_snap, dtype=torch.long)

                    if args.pre_training == "dont_use":
                        if args.exp_name == "exp_2":
                            joint_n, ngh_and_batch_id_n, joint_p, ngh_and_batch_id_p = modelN.contrast(args.exp_name, args.pre_training, pos_index[0], pos_index[1], neg_dst, time_snap, edge_snap, test =True)

                            _, ngh_and_batch_id_p = torch.unique(ngh_and_batch_id_p[:,1], return_inverse=True)
                            _, ngh_and_batch_id_n = torch.unique(ngh_and_batch_id_n[:,1], return_inverse=True)
                            pos_hop = scatter(joint_p, ngh_and_batch_id_p, dim=0, reduce="mean")
                            neg_hop = scatter(joint_n, ngh_and_batch_id_n, dim=0, reduce="mean")
                        else:
                            pos_hop , neg_hop = modelN.contrast(args.exp_name , args.pre_training, pos_index[0], pos_index[1], neg_dst, time_snap, edge_snap)  
                    if args.pre_training == "use_pretrain":
                        if args.exp_name == "exp_2":
                            joint_n, ngh_and_batch_id_n, joint_p, ngh_and_batch_id_p = modelN.contrast(args.exp_name, args.pre_training, pos_index[0], pos_index[1], neg_dst, time_snap, edge_snap, test =True)

                            _, ngh_and_batch_id_p = torch.unique(ngh_and_batch_id_p[:,1], return_inverse=True)
                            _, ngh_and_batch_id_n = torch.unique(ngh_and_batch_id_n[:,1], return_inverse=True)
                            pos_hop = scatter(joint_p, ngh_and_batch_id_p, dim=0, reduce="mean")
                            neg_hop = scatter(joint_n, ngh_and_batch_id_n, dim=0, reduce="mean")
                            # print("pos_hop", pos_hop.shape)
                            # print("h[pos_index[0]]", h[pos_index[0]].shape)
                            # print("neg_hop", neg_hop.shape, ngh_and_batch_id_n[:,1], uni, uni.shape)
                            # print("h[neg_dst]", h[neg_dst].shape)
                        elif args.exp_name == "exp_1":
                            modelN.set_seed(1)
                            pos_hop = None
                            neg_hop = None
                            for hop in range(args.n_hop):
                                hop = int(hop)
                                # print("hping", pos_index[0] , args.n_degree[hop] , modelN.ncache_hash(pos_index[1], hop+1))
                                idx_pos = pos_index[0] * int(args.n_degree[hop]) + modelN.ncache_hash(pos_index[1], hop+1)
                                idx_neg = pos_index[0] * int(args.n_degree[hop]) + modelN.ncache_hash(neg_dst, hop+1)
                                vec_pos= modelN.get_neighborhood_store()[hop+1][idx_pos,3:]
                                vec_neg = modelN.get_neighborhood_store()[hop+1][idx_neg,3:]
                                if pos_hop is None:
                             
                                    pos_hop = vec_pos.clone()
                                    neg_hop = vec_neg.clone()
                                else:
                                    pos_hop = torch.cat([pos_hop, vec_pos], dim=-1)  # concat along last dim
                                    neg_hop = torch.cat([neg_hop, vec_neg], dim=-1)
                                
                        else:
                            pos_hop , neg_hop = modelN.contrast(args.exp_name, args.pre_training, pos_index[0], pos_index[1], neg_dst, time_snap, edge_snap, test =True)
                    
                        
                  
                    cat_min_true = pos_hop.min()
                    cat_max_true = pos_hop.max()
                    pos_hop = (pos_hop - cat_min_true) / (cat_max_true - cat_min_true)

                    cat_min_fake = neg_hop.min()
                    cat_max_fake = neg_hop.max()
                    neg_hop = (neg_hop - cat_min_fake) / (cat_max_fake - cat_min_fake)
                    # print("pos_hop", pos_hop.shape)
                    pos_out = link_pred(h[pos_index[0]], h[pos_index[1]],pos_hop)
                    neg_out = link_pred(h[pos_index[0]], h[neg_dst], neg_hop)

                    if k <2:
                        print("pos_hop", pos_hop.mean(dim=0, keepdim=True).mean(), pos_hop.mean(dim=0, keepdim=True).std())
                        print("neg_hop", neg_hop.mean(dim=0, keepdim=True).mean(), neg_hop.mean(dim=0, keepdim=True).std())
           
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


          
            evaluator = Evaluator(name=args.dataset)
            neg_sampler = dataset.negative_sampler
            if args.with_hop ==1:
                NAT_module.logger.info(f"train_loss: {total_loss} train time: {train_time} \n")
            if (args.wandb):
                wandb.log({"train_loss":(total_loss),
                        "train time": train_time,
                        })
            writer.add_scalar("Loss/train", total_loss, epoch)
            
            if epoch % 49 == 0:
                dataset.load_test_ns()
                test_snapshots = data['test_data']['edge_index']
                ts_list = data['test_data']['ts_map']
                test_loader = IndexedTemporalDataLoader(test_edges, batch_size=batch_size)
                neg_sampler = dataset.negative_sampler
                dataset.load_test_ns()

                test_start_time = timeit.default_timer()
                if args.with_hop ==1:
                    test_metrics, h, h_0, c_0 = test_tgb(args.exp_name, full_data, interval,random_sampler_test,h, h_0, c_0, test_loader, test_snapshots, ts_list,
                                                         node_feat,model, link_pred,neg_sampler,evaluator,metric, modelN, split_mode='test')
                if args.with_hop ==0:
                    test_metrics, h, h_0, c_0 = test_tgb(args.exp_name, full_data, interval,random_sampler_test,h, h_0, c_0, test_loader, test_snapshots, ts_list,
                                                         node_feat,model, link_pred,neg_sampler,evaluator,metric, split_mode='test')
                test_time = timeit.default_timer() - test_start_time
                if test_metrics > best_test:
                    best_test = test_metrics
                    best_epoch = epoch
                if test_time < best_test_time:
                   best_test_time = test_time    
                if args.with_hop ==1:
                    NAT_module.logger.info(f"best epoch {best_epoch} \n, test_metrics {test_metrics} \n, test_time {test_time}\n")
                writer.add_scalar("MRR/test", test_metrics, epoch)

                print ("test metric is ", test_metrics)
                print ("test elapsed time is ", test_time)
                print ("--------------------------------")
                
                
        print ("run finishes")
        print ("best epoch is, ", best_epoch)
        print ("best test performance is, ", best_test)
        print("best timings ", best_test_time)
        print ("------------------------------------------")
        if args.with_hop ==1:
            NAT_module.logger.info(f"best epoch is, {best_epoch} ,best test performance is {best_test} , best timings is {best_test_time}")
        runs_best_test.append(best_test)
        runs_best_times.append(best_test_time)
        
    runs_best_test = np.array(runs_best_test)
    runs_best_times = np.array(runs_best_times)

    test_mean, test_std = runs_best_test.mean(), runs_best_test.std(ddof=1)
    test_mean_time, test_std_time = runs_best_times.mean(), runs_best_times.std(ddof=1)
    
    if args.with_hop ==1:
        NAT_module.logger.info(f"Test MRR: mean={test_mean:.4f} ± {test_std:.4f} ; Test time: mean={test_mean_time:.4f} ± {test_std_time:.4f}")
    print("========================================")
    print(f"Summary of {args.num_runs} runs:")
    print(f"Test MRR: mean={test_mean:.4f} ± {test_std:.4f}")
    print(f"Test time: mean={test_mean_time:.4f} ± {test_std_time:.4f}")

    print("========================================")
    
    writer.close()
