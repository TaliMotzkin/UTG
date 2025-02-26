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
    def __init__(self, train_edges,val_edges, test_edges, args):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(init_nat_module, self).__init__()
        self.NUM_NEIGHBORS = args.n_degree
        self.ATTN_NUM_HEADS = args.attn_n_head
        self.DROP_OUT = args.nat_drop_out
        self.train = train_edges
        self.test = test_edges
        self.val = val_edges
        self.NUM_HOP = args.n_hop
        self.POS_DIM = args.pos_dim
        self.VERBOSITY = args.verbosity
        self.SEED = args.seed
        self.TIME_DIM = args.time_dim
        self.REPLACE_PROB = args.replace_prob
        self.SELF_DIM = args.self_dim
        self.NGH_DIM = args.ngh_dim
        self.set_nat()



    def set_nat(self, x, edge_index, edge_weight, h, c):


        assert(self.NUM_HOP < 3) # only up to second hop is supported
        set_random_seed(self.SEED)
        logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, best_model_ngh_store_path = set_up_logger(args, sys_argv)



        # multiprocessing memory setting
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

        
  feat_dim = n_feat.shape[1]
  e_feat_dim = e_feat.shape[1]
  time_dim = TIME_DIM
  model_dim = feat_dim + e_feat_dim + time_dim
  hidden_dim = e_feat_dim + time_dim
  num_raw = 3
  memory_dim = NGH_DIM + num_raw
  num_neighbors = [1]
  for i in range(NUM_HOP):
    num_neighbors.extend([int(NUM_NEIGHBORS[i])])
  # num_neighbors.extend([int(n) for n in NUM_NEIGHBORS]) # the 0-hop neighborhood has only 1 node

  total_start = time.time()
  nat = NAT(n_feat, e_feat, memory_dim, max_idx + 1, time_dim=TIME_DIM, pos_dim=POS_DIM, n_head=ATTN_NUM_HEADS, num_neighbors=num_neighbors, dropout=DROP_OUT,
    linear_out=args.linear_out, get_checkpoint_path=get_checkpoint_path, get_ngh_store_path=get_ngh_store_path, get_self_rep_path=get_self_rep_path, get_prev_raw_path=get_prev_raw_path, verbosity=VERBOSITY,
  n_hops=NUM_HOP, replace_prob=REPLACE_PROB, self_dim=SELF_DIM, ngh_dim=NGH_DIM, device=device)
  nat.to(device)
  nat.reset_store()

  # print("NAT tali", len(nat.get_neighborhood_store()), nat.get_neighborhood_store()[0][0], nat.get_neighborhood_store()[1][0:32], nat.get_neighborhood_store()[2][3]) 
  optimizer = torch.optim.Adam(nat.parameters(), lr=LEARNING_RATE)
  criterion = torch.nn.BCELoss()
  early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

  # start train and val phases
  train_val(train_val_data, nat, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, rand_samplers, logger, model_dim, n_hop=NUM_HOP)

  # print("NAT tali after train", len(nat.get_neighborhood_store()), nat.get_neighborhood_store()[0][0], nat.get_neighborhood_store()[1][0:32], nat.get_neighborhood_store()[2][3]) 
  # final testing
  print("_*"*50)
  if args.mode == 'i':
    nat.reset_store()
    nat.reset_self_rep()
    train_acc, train_ap, train_f1, train_auc = eval_one_epoch('test for {} nodes'.format(args.mode), nat, all_train_val_rand_sampler, all_train_val_src_l, all_train_val_tgt_l, all_train_val_ts_l, all_train_val_label_l, all_train_val_e_idx_l, bs=32)
  test_start = time.time()
  test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), nat, test_rand_sampler, test_src_l, test_tgt_l, test_ts_l, test_label_l, test_e_idx_l)
  test_end = time.time()
  logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}, time: {}'.format(args.mode, test_acc, test_auc, test_ap, test_end - test_start))
  test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
  if args.mode == 'i':
    inductive_auc.append(test_auc)
    inductive_ap.append(test_ap)
  else:
    transductive_auc.append(test_auc)
    transductive_ap.append(test_ap)
  test_times.append(test_end - test_start)
  early_stoppers.append(early_stopper.best_epoch + 1)
  # save model
  logger.info('Saving NAT model ...')
  torch.save(nat.state_dict(), best_model_path)
  logger.info('NAT model saved')

  # save one line result
  save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
  # save walk_encodings_scores
  total_end = time.time()
  print("NAT experiment statistics:")
  if args.mode == "t":
    nat_results(logger, transductive_auc, "transductive_auc")
    nat_results(logger, transductive_ap, "transductive_ap")
  else:
    nat_results(logger, inductive_auc, "inductive_auc")
    nat_results(logger, inductive_ap, "inductive_ap")
  
  nat_results(logger, test_times, "test_times")
  nat_results(logger, early_stoppers, "early_stoppers")
  total_time.append(total_end - total_start)
  nat_results(logger, total_time, "total_time")