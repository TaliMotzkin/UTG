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
        return nat

    def contrast_nat(self, src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut):

        pos_prob, neg_prob = self.nat.contrast(src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut)


        
        # print("NAT tali", len(nat.get_neighborhood_store()), nat.get_neighborhood_store()[0][0], nat.get_neighborhood_store()[1][0:32], nat.get_neighborhood_store()[2][3]) 
##################maybe dont need the rest - check! ######################

  # start train and val phases
  # train_val(train_val_data, nat, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, rand_samplers, logger, model_dim, n_hop=NUM_HOP)

  # # print("NAT tali after train", len(nat.get_neighborhood_store()), nat.get_neighborhood_store()[0][0], nat.get_neighborhood_store()[1][0:32], nat.get_neighborhood_store()[2][3]) 
  # # final testing
  # print("_*"*50)
  # if args.mode == 'i':
  #   nat.reset_store()
  #   nat.reset_self_rep()
  #   train_acc, train_ap, train_f1, train_auc = eval_one_epoch('test for {} nodes'.format(args.mode), nat, all_train_val_rand_sampler, all_train_val_src_l, all_train_val_tgt_l, all_train_val_ts_l, all_train_val_label_l, all_train_val_e_idx_l, bs=32)
  # test_start = time.time()
  # test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), nat, test_rand_sampler, test_src_l, test_tgt_l, test_ts_l, test_label_l, test_e_idx_l)
  # test_end = time.time()
  # logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}, time: {}'.format(args.mode, test_acc, test_auc, test_ap, test_end - test_start))
  # test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
  # if args.mode == 'i':
  #   inductive_auc.append(test_auc)
  #   inductive_ap.append(test_ap)
  # else:
  #   transductive_auc.append(test_auc)
  #   transductive_ap.append(test_ap)
  # test_times.append(test_end - test_start)
  # early_stoppers.append(early_stopper.best_epoch + 1)
  # # save model
  # logger.info('Saving NAT model ...')
  # torch.save(nat.state_dict(), best_model_path)
  # logger.info('NAT model saved')

  # # save one line result
  # save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
  # # save walk_encodings_scores
  # total_end = time.time()
  # print("NAT experiment statistics:")
  # if args.mode == "t":
  #   nat_results(logger, transductive_auc, "transductive_auc")
  #   nat_results(logger, transductive_ap, "transductive_ap")
  # else:
  #   nat_results(logger, inductive_auc, "inductive_auc")
  #   nat_results(logger, inductive_ap, "inductive_ap")
  
  # nat_results(logger, test_times, "test_times")
  # nat_results(logger, early_stoppers, "early_stoppers")
  # total_time.append(total_end - total_start)
  # nat_results(logger, total_time, "total_time")
