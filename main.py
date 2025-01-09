#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import dgl
import math
import wandb
import torch
import random
import argparse
import numpy as np
import os

from topo_utils.image_utils import compute_topo_features

from tqdm import tqdm
from model import WinGNN
from test_new import test
from train_new import train
from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_r
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset
import yaml
import warnings
warnings.filterwarnings("ignore")

def dict_to_namespace(d):
    """将嵌套字典转换为嵌套的Namespace对象"""
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 将yaml配置转换为嵌套的命名空间
    args = dict_to_namespace(config)

    logger = getLogger(cfg.log_path)

    # 创建模型保存目录
    if not os.path.exists('model_parameter'):
        os.makedirs('model_parameter')

    # load datasets
    if args.dataset == 'dblp':
        dataset = args.dataset
        e_feat = np.load('dataset/{0}/ml_{0}.npy'.format(dataset))
        n_feat_ = np.load('dataset/{0}/ml_{0}_node.npy'.format(dataset))
        train_data, train_e_feat, train_n_feat, test_data, test_e_feat, test_n_feat = load("Norandom", len(n_feat_))
        graphs = []
        for tr in train_data:
            graphs.append(tr)
        for te in test_data:
            graphs.append(te)
        n_feat = [n_feat_ for i in range(len(graphs))]
    elif args.dataset in ["reddit-body", "reddit-title", "as",
                          "uci-msg", "bitcoin-otc", "bitcoin-alpha",
                          'stackoverflow_M']:
        graphs, e_feat, e_time, n_feat = load_r(args.dataset)
    else:
        raise ValueError

    n_dim = n_feat[0].shape[1]
    n_node = n_feat[0].shape[0]


    if args.cuda_device >= 0:
        if not torch.cuda.is_available():
            print("CUDA不可用，使用CPU替代")
            device = torch.device('cpu')
        elif args.cuda_device >= torch.cuda.device_count():
            print(f"GPU {args.cuda_device} 不可用，使用GPU 0替代")
            device = torch.device('cuda:0')
        else:
            device = torch.device(f'cuda:{args.cuda_device}')
    else:
        device = torch.device('cpu')

    all_mrr_avg = []
    all_auc_avg = []
    all_rl1_avg = []
    all_rl3_avg = []
    all_rl10_avg = []
    all_acc_avg = []
    all_ap_avg = []
    all_f1_avg = []
    all_macro_auc_avg = []
    all_micro_auc_avg = []
    best_mrr = 0.0
    best_model = 0

    for rep in range(args.repeat):

        logger.info('num_layers:{}, num_hidden: {}, lr: {}, maml_lr:{}, window_num:{}, drop_rate:{}, 负样本采样固定'.
                    format(args.num_layers, args.num_hidden, args.lr, args.maml_lr, args.window_num, args.drop_rate))
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        graph_l = []
        # Data set processing
        for idx, graph in tqdm(enumerate(graphs)):
            graph_d = dgl.from_scipy(graph)

            graph_d.edge_feature = torch.Tensor(e_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])

            if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
                n_feat_t = graph_l[idx - 1].node_feature
                graph_d.node_feature = torch.Tensor(n_feat_t)
            else:
                graph_d.node_feature = torch.Tensor(n_feat[idx])

            graph_d = dgl.remove_self_loop(graph_d)
            graph_d = dgl.add_self_loop(graph_d)

            edges = graph_d.edges()
            row = edges[0].numpy()
            col = edges[1].numpy()
            # Negative sample sampling 1:1
            n_e = graph_d.num_edges() - graph_d.num_nodes()
            # Edge label
            y_pos = np.ones(shape=(n_e,))
            y_neg = np.zeros(shape=(n_e,))
            y = list(y_pos) + list(y_neg)

            edge_label_index = list()
            edge_label_index.append(row.tolist()[:n_e])
            edge_label_index.append(col.tolist()[:n_e])

            graph_d.edge_label = torch.Tensor(y)
            graph_d.edge_label_index = torch.LongTensor(edge_label_index)

            graph_l.append(graph_d)
        # Negative sample sampling 1:1
        for idx, graph in tqdm(enumerate(graphs)):
            graph = Graph(
                node_feature=graph_l[idx].node_feature,
                edge_feature=graph_l[idx].edge_feature,
                edge_index=graph_l[idx].edge_label_index,
                edge_time=graph_l[idx].edge_time,
                directed=True
            )

            dataset = GraphDataset(graph,
                                   task='link_pred',
                                   edge_negative_sampling_ratio=1.0,
                                   minimum_node_per_graph=5)
            edge_labe_index = dataset.graphs[0].edge_label_index
            graph_l[idx].edge_label_index = torch.LongTensor(edge_labe_index)
        
        topo_features = compute_topo_features(args)
        for i in range(len(graph_l)):
            graph_l[i].topo_feature = topo_features[i]
        # model initialization
        model = WinGNN.Model(n_dim, args.out_dim, args.num_hidden, args.num_layers, args.dropout, args)
        model.train()

        # LightDyG optimizer
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

        model = model.to(device)

        model_save_path = 'model_parameter/' + args.dataset

        model_load_path = 'model_parameter/' + args.dataset

        # It is divided into multiple Windows,
        # each of which is meta updated in addition to the meta training window

        # Partition dataset
        n = math.ceil(len(graph_l) * 0.7)

        # train
        best_param = train(args, model, optimizer, device, graph_l, logger, n)

        model.load_state_dict(best_param['best_state'])
        S_dw = best_param['best_s_dw']

        # test
        model.eval()
        avg_mrr, avg_auc, rl1_avg, rl3_avg, rl10_avg, acc_avg, ap_avg, f1_avg, macro_auc_avg, micro_auc_avg = test(graph_l, model, args, logger, n, S_dw, device)

        if avg_mrr > best_mrr:
            best_model = best_param['best_state']
        all_mrr_avg.append(avg_mrr)
        all_auc_avg.append(avg_auc)
        all_rl1_avg.append(rl1_avg)
        all_rl3_avg.append(rl3_avg)
        all_rl10_avg.append(rl10_avg)
        all_acc_avg.append(acc_avg)
        all_ap_avg.append(ap_avg)
        all_f1_avg.append(f1_avg)
        all_macro_auc_avg.append(macro_auc_avg)
        all_micro_auc_avg.append(micro_auc_avg)
    torch.save(best_model, model_save_path + '.pkl')
    # 计算平均值和标准差
    results = {
        'mrr': {'mean': np.mean(all_mrr_avg), 'std': np.std(all_mrr_avg)},
        'auc': {'mean': np.mean(all_auc_avg), 'std': np.std(all_auc_avg)},
        'rl1': {'mean': np.mean(all_rl1_avg), 'std': np.std(all_rl1_avg)},
        'rl3': {'mean': np.mean(all_rl3_avg), 'std': np.std(all_rl3_avg)}, 
        'rl10': {'mean': np.mean(all_rl10_avg), 'std': np.std(all_rl10_avg)},
        'acc': {'mean': np.mean(all_acc_avg), 'std': np.std(all_acc_avg)},
        'ap': {'mean': np.mean(all_ap_avg), 'std': np.std(all_ap_avg)},
        'f1': {'mean': np.mean(all_f1_avg), 'std': np.std(all_f1_avg)},
        'macro_auc': {'mean': np.mean(all_macro_auc_avg), 'std': np.std(all_macro_auc_avg)},
        'micro_auc': {'mean': np.mean(all_micro_auc_avg), 'std': np.std(all_micro_auc_avg)}
    }
    print(results)
    # 生成时间戳文件夹名
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H.%M')
    save_dir = f'results/{args.dataset}/{timestamp}_{args.topo.use_topo}'
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存结果到json文件
    import json
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(all_mrr_avg)

