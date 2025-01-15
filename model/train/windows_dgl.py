#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import torch
import random
import datetime
import logging
import numpy as np
from copy import deepcopy
from model.loss import prediction
from model.utils import report_rank_based_eval_meta
import datetime
import logging
import numpy as np
import torch
from tqdm import tqdm
import logging
from utils.config import cfg
from model.loss import compute_loss, compute_loss_meta
from utils.config import makedirs_rm_exist
from torch.utils.tensorboard import SummaryWriter

def windows_train(model, optimizer, graph_l, writer):
    device = next(model.parameters()).device
    n = len(graph_l)
    best_param = {'best_mrr': 0, 'best_state': None, 'best_s_dw': None}
    earl_stop_c = 0
    epoch_count = 0

    for epoch in tqdm(range(cfg.optim.max_epoch)):
        # Keep a version of the data without gradient calculation
        graph_l_cpy = deepcopy(graph_l)
        all_mrr = 0.0
        i = 0
        fast_weights = [w.to(device) for w in model.parameters()]
        S_dw = [0] * len(fast_weights)
        train_count = 0
        # LightDyG windows calculation
        while i < (n - cfg.windows.window_size):
            if i != 0:
                i = random.randint(i, i + cfg.windows.window_size)
            if i >= (n - cfg.windows.window_size):
                break
            graph_train = graph_l[i: i + cfg.windows.window_size]
            i = i + 1
            # Copy a version of data as a valid in the window
            features = [graph_unit.node_feature.to(device) for graph_unit in graph_train]
            
            # 确保fast_weights在正确的设备上
            fast_weights = [w.to(device) for w in model.parameters()]
            
            window_mrr = 0.0
            losses = torch.tensor(0.0).to(device)
            count = 0
            # one window
            for idx, graph in enumerate(graph_train):
                # The last snapshot in the window is valid only
                if idx == cfg.windows.window_size - 1:
                    break
                # t snapshot train
                # Copy a version of data as a train in the window
                feature_train = features[idx].to(device)
                graph = graph.to(device)
                pred = model(graph, feature_train, fast_weights)
                loss = compute_loss_meta(pred, graph.edge_label)

                # t grad
                grad = torch.autograd.grad(loss, fast_weights)

                graph = graph.to('cpu')
                feature_train = feature_train.to('cpu')

                beta = cfg.windows.beta
                S_dw = [beta * s + (1 - beta) * g * g for s, g in zip(S_dw, grad)]
                fast_weights = [w - cfg.windows.maml_lr / (torch.sqrt(s) + 1e-8) * g 
                              for w, g, s in zip(fast_weights, grad, S_dw)]

                # t+1 snapshot valid
                graph_train[idx + 1] = graph_train[idx + 1].to(device)
                pred = model(graph_train[idx + 1], features[idx + 1], fast_weights)
                loss = compute_loss_meta(pred, graph_train[idx + 1].edge_label)

                edge_label = graph_train[idx + 1].edge_label
                edge_label_index = graph_train[idx + 1].edge_label_index
                mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_train[idx + 1], features[idx+1],
                                                                  fast_weights)
                graph_train[idx + 1].edge_label = edge_label
                graph_train[idx + 1].edge_label_index = edge_label_index

                droprate = torch.FloatTensor(np.ones(shape=(1)) * cfg.windows.drop_rate)
                masks = torch.bernoulli(1. - droprate).unsqueeze(1)
                if masks[0][0]:
                    losses = losses + loss
                    count += 1
                    window_mrr += mrr
                acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_train[idx + 1].edge_label)
                perf = {'loss': loss.item(), 'mrr': mrr, 'acc': acc, 'ap': ap, 'f1': f1, 
                       'macro_auc': macro_auc, 'micro_auc': micro_auc}
                writer.add_scalars('train', perf, epoch)

            if losses:
                losses = losses / count
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            if count:
                all_mrr += window_mrr / count
            train_count += 1

        all_mrr = all_mrr / train_count
        epoch_count += 1

        if all_mrr > best_param['best_mrr']:
            best_param = {'best_mrr': all_mrr, 'best_state': deepcopy(model.state_dict()), 'best_s_dw': S_dw}
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break
    return best_param

def windows_test(graph_l, model, S_dw):
    device = next(model.parameters()).device
    n = len(graph_l)
    beta = cfg.windows.beta
    avg_mrr = 0.0
    avg_auc = 0.0
    rl1_avg = 0.0
    rl3_avg = 0.0
    rl10_avg = 0.0
    acc_avg = 0.0
    ap_avg = 0.0
    f1_avg = 0.0
    macro_auc_avg = 0.0
    micro_auc_avg = 0.0

    graph_test = graph_l
    # model parameters
    fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
    for idx, g_test in tqdm(enumerate(graph_test)):

        if idx == len(graph_test) - 1:
            break

        graph_train = deepcopy(g_test.node_feature)
        graph_train = graph_train.to(device)
        g_test = g_test.to(device)

        pred = model(g_test, graph_train, fast_weights)
        loss = compute_loss_meta(pred, g_test.edge_label)

        graph_train = graph_train.to('cpu')
        grad = torch.autograd.grad(loss, fast_weights)
        g_test = g_test.to('cpu')

        S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0].pow(2), zip(grad, S_dw)))

        fast_weights = list(map(lambda p: p[1] - cfg.windows.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

        graph_test[idx + 1] = graph_test[idx + 1].to(device)
        graph_test[idx + 1].node_feature = graph_test[idx + 1].node_feature.to(device)
        pred = model(graph_test[idx + 1], graph_test[idx + 1].node_feature, fast_weights)

        loss = compute_loss_meta(pred, graph_test[idx + 1].edge_label)

        edge_label = graph_test[idx + 1].edge_label
        edge_label_index = graph_test[idx + 1].edge_label_index
        mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_test[idx + 1], graph_test[idx + 1].node_feature,
                                                          fast_weights)
        graph_test[idx + 1].edge_label = edge_label
        graph_test[idx + 1].edge_label_index = edge_label_index

        acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_test[idx + 1].edge_label)
        avg_mrr += mrr
        avg_auc += macro_auc
        rl1_avg += rl1
        rl3_avg += rl3
        rl10_avg += rl10
        acc_avg += acc
        ap_avg += ap
        f1_avg += f1
        macro_auc_avg += macro_auc
        micro_auc_avg += micro_auc

    avg_mrr /= len(graph_test) - 1
    avg_auc /= len(graph_test) - 1
    rl1_avg /= len(graph_test) - 1
    rl3_avg /= len(graph_test) - 1
    rl10_avg /= len(graph_test) - 1
    acc_avg /= len(graph_test) - 1
    ap_avg /= len(graph_test) - 1
    f1_avg /= len(graph_test) - 1
    macro_auc_avg /= len(graph_test) - 1
    micro_auc_avg /= len(graph_test) - 1
    return {'mrr': avg_mrr, 'avg_auc': avg_auc, 'rck1': rl1_avg, 'rck3': rl3_avg,
            'rck10': rl10_avg, 'acc': acc_avg, 'ap': ap_avg, 'f1': f1_avg, 'macro_auc': macro_auc_avg, 'micro_auc': micro_auc_avg}

def train_windows_dgl(loggers, model, optimizer, scheduler, datasets,
                      **kwargs):
    num_splits = len(loggers)
    
    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    out_dir = cfg.out_dir
    print(f'Tensorboard directory: {out_dir}')
    makedirs_rm_exist(f'./{out_dir}')
    writer = SummaryWriter(f'./{out_dir}')
    with open(f'./{out_dir}/config.yaml', 'w') as f:
        cfg.dump(stream=f)
        

    model.train()
    best_param = windows_train(model, optimizer, datasets[0], writer)
    model.load_state_dict(best_param['best_state'])
    S_dw = best_param['best_s_dw']
    model.eval()
    perf = windows_test(datasets[1], model, S_dw)
    writer.add_scalars('test', perf)
    
    logging.info('Task done, results saved in {}'.format(cfg.out_dir))