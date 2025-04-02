import pandas as pd

import resource
import torch.nn as nn
import statistics
import os
import sys
from log import *


project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'TGX'))
sys.path.insert(0, os.path.join(project_root, 'NAT'))

from NAT.parser import *
from NAT.eval import *
from NAT.utils import *
from NAT.train import *
from NAT.module import NAT

class init_nat_module():
    def __init__(self,full_data, e_feat, n_feat, train_edges,val_edges, test_edges, args, sys_argv):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(init_nat_module, self).__init__()
        self.args = args
        self.sys_argv = sys_argv
        self.NUM_NEIGHBORS = args.n_degree
        self.ATTN_NUM_HEADS = args.attn_n_head
        self.DROP_OUT = args.nat_drop_out
        self.train = train_edges
        self.test = test_edges
        self.val = val_edges
        self.full_data = full_data
        self.e_feat = e_feat.cpu().numpy()
        self.n_feat = n_feat.cpu().numpy()
        self.NUM_HOP = args.n_hop
        self.POS_DIM = args.pos_dim
        self.VERBOSITY = args.verbosity
        self.SEED = args.seed
        self.TIME_DIM = args.time_dim
        self.REPLACE_PROB = args.replace_prob
        self.SELF_DIM = args.self_dim
        self.NGH_DIM = args.ngh_dim
        self.linear_out = args.linear_out
        self.device = args.device
        self.nat = self.set_nat()
        



    def set_nat(self):

        max_idx = max( self.full_data.src.max(),  self.full_data.dst.max())
        assert(np.unique(np.stack([self.full_data.src.cpu() , self.full_data.dst.cpu() ])).shape[0] == max_idx +1)  # all nodes except node 0 should appear and be compactly indexed
        assert(self.n_feat.shape[0] == max_idx + 1) 


        assert(self.NUM_HOP < 3) # only up to second hop is supported
        set_random_seed(self.SEED)
        logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, best_model_ngh_store_path = set_up_logger(self.args, self.sys_argv)


        # multiprocessing memory setting
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (200*self.args.batch_size, rlimit[1]))

        feat_dim = self.n_feat.shape[1]
        e_feat_dim = self.e_feat.shape[1]

        time_dim = self.TIME_DIM
        model_dim = feat_dim + e_feat_dim + time_dim
        
        hidden_dim = e_feat_dim + time_dim
        num_raw = 3
        memory_dim = self.NGH_DIM + num_raw
        
        num_neighbors = [1]
        for i in range(self.NUM_HOP):
            num_neighbors.extend([int(self.NUM_NEIGHBORS[i])])
      # num_neighbors.extend([int(n) for n in NUM_NEIGHBORS]) # the 0-hop neighborhood has only 1 node
    
        total_start = time.time()
        nat = NAT(self.n_feat, self.e_feat, memory_dim, max_idx + 1, time_dim=self.TIME_DIM, pos_dim=self.POS_DIM, n_head=self.ATTN_NUM_HEADS,
                  num_neighbors=num_neighbors, dropout=self.DROP_OUT, linear_out=self.linear_out, get_checkpoint_path=get_checkpoint_path,
                  get_ngh_store_path=get_ngh_store_path, get_self_rep_path=get_self_rep_path, get_prev_raw_path=get_prev_raw_path, verbosity=self.VERBOSITY,
                  n_hops=self.NUM_HOP, replace_prob=self.REPLACE_PROB, self_dim=self.SELF_DIM, ngh_dim=self.NGH_DIM, device=self.device)
        nat.to(self.device)
        nat.reset_store()
        self.logger = logger
        return nat


    def contrast_nat(self, pre_training, src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut, test = False):

        pos_prob, neg_prob = self.nat.contrast(pre_training, src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut, test)
        return pos_prob, neg_prob 


