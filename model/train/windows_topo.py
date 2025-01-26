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
from model.train.wingnn_train_utils import report_rank_based_eval_meta
import datetime
import logging
import numpy as np
import torch
from tqdm import tqdm
import logging
from utils.config import cfg
from model.loss import compute_loss_meta
from utils.config import makedirs_rm_exist
from torch.utils.tensorboard import SummaryWriter


def generate_learning_rates(meta_model, features):
    meta_model.train()
    features = features.to(torch.device(cfg.device))
    learning_rates = meta_model(features)
    # 使用sigmoid确保学习率为正且在合理范围内
    # 使用ReLU+clamp的组合来限制学习率范围
    # ReLU确保学习率非负,clamp进一步限制上界
    # 这样可以避免sigmoid在接近0和1时梯度消失的问题
    learning_rates = torch.sigmoid(learning_rates) * cfg.windows.maml_lr  # 将学习率限制在0到0.1之间
    # learning_rates = torch.clamp(torch.relu(learning_rates), min=1e-4, max=1.0)
    return learning_rates

def windows_train(model,meta_model, optimizer, meta_optimizer, graph_l, writer, topo_features):
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
        S_dw = [0] * len([param for param in model.parameters() if param.requires_grad])
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
            
            window_mrr = 0.0
            losses = []  # 使用列表存储所有损失
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
                
                # 第一次前向传播
                pred = model(graph, feature_train)
                loss = compute_loss_meta(pred, graph.edge_label)
                
                # 计算梯度但不更新参数
                grad = torch.autograd.grad(loss, [param for param in model.parameters() if param.requires_grad])
                
                graph = graph.to('cpu')
                feature_train = feature_train.to('cpu')
                
                # 更新S_dw
                beta = cfg.windows.beta
                S_dw = [beta * s + (1 - beta) * g * g for s, g in zip(S_dw, grad)]
                
                # 手动更新参数
                learning_rates = generate_learning_rates(meta_model, topo_features[idx] - topo_features[idx - 1])
                with torch.no_grad():  # 使用no_grad来避免创建新的计算图
                    for param, g, s, lr in zip([param for param in model.parameters() if param.requires_grad], grad, S_dw, learning_rates):
                        param.data = param.data - lr * g
                
                # 第二次前向传播
                graph_train[idx + 1] = graph_train[idx + 1].to(device)
                pred = model(graph_train[idx + 1], features[idx + 1])
                current_loss = compute_loss_meta(pred, graph_train[idx + 1].edge_label)
                
                # 保存当前的edge_label和edge_label_index
                edge_label = graph_train[idx + 1].edge_label
                edge_label_index = graph_train[idx + 1].edge_label_index
                
                # 计算评估指标
                mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_train[idx + 1], features[idx+1],
                                                                  [param for param in model.parameters() if param.requires_grad])
                
                # 恢复edge_label和edge_label_index
                graph_train[idx + 1].edge_label = edge_label
                graph_train[idx + 1].edge_label_index = edge_label_index
                
                mask = torch.bernoulli(torch.tensor(1. - cfg.windows.drop_rate))
                if mask.item():
                    losses.append(current_loss)
                    count += 1
                    window_mrr += mrr
                
            if losses and count > 0:
                # 计算总损失
                total_loss = torch.stack(losses).mean()
                
                # 更新模型参数
                optimizer.zero_grad()
                meta_optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                meta_optimizer.step()
                
                # 清除计算图
                for loss in losses:
                    loss.detach_()
            if count:
                all_mrr += window_mrr / count
            train_count += 1

        all_mrr = all_mrr / train_count
        epoch_count += 1

        if all_mrr > best_param['best_mrr']:
            best_param = {'best_mrr': all_mrr, 'best_state': deepcopy(model.state_dict()), 'best_s_dw': S_dw, 'best_meta_state': deepcopy(meta_model.state_dict())}
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break
    return best_param

def windows_test(graph_test, model, meta_model, S_dw, topo_features):
    device = next(model.parameters()).device
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
    # model parameters
    for idx, g_test in tqdm(enumerate(graph_test)):

        if idx == len(graph_test) - 1:
            break

        graph_train = deepcopy(g_test.node_feature)
        graph_train = graph_train.to(device)
        g_test = g_test.to(device)

        pred = model(g_test, graph_train)
        loss = compute_loss_meta(pred, g_test.edge_label)

        graph_train = graph_train.to('cpu')
        grad = torch.autograd.grad(loss, [param for param in model.parameters() if param.requires_grad])
        g_test = g_test.to('cpu')

        S_dw = [beta * s + (1 - beta) * g.pow(2) for g, s in zip(grad, S_dw)]

        learning_rates = generate_learning_rates(meta_model, topo_features[idx] - topo_features[idx - 1])
        for param, g, s, lr in zip([param for param in model.parameters() if param.requires_grad], grad, S_dw, learning_rates):
            param.data = param.data - lr * g

        graph_test[idx + 1] = graph_test[idx + 1].to(device)
        graph_test[idx + 1].node_feature = graph_test[idx + 1].node_feature.to(device)
        pred = model(graph_test[idx + 1], graph_test[idx + 1].node_feature)

        loss = compute_loss_meta(pred, graph_test[idx + 1].edge_label)

        edge_label = graph_test[idx + 1].edge_label
        edge_label_index = graph_test[idx + 1].edge_label_index
        mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_test[idx + 1], graph_test[idx + 1].node_feature,
                                                          [param for param in model.parameters() if param.requires_grad])
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
    # 检查所有指标是否存在nan值
    metrics = {
        'mrr': avg_mrr, 
        'avg_auc': avg_auc,
        'rck1': rl1_avg,
        'rck3': rl3_avg, 
        'rck10': rl10_avg,
        'acc': acc_avg,
        'ap': ap_avg,
        'f1': f1_avg,
        'macro_auc': macro_auc_avg,
        'micro_auc': micro_auc_avg
    }
    
    for metric_name, value in metrics.items():
        if torch.isnan(value) if torch.is_tensor(value) else np.isnan(value):
            logging.warning(f'警告: {metric_name} 的值为 NaN')
    return {'mrr': avg_mrr, 'avg_auc': avg_auc, 'rck1': rl1_avg, 'rck3': rl3_avg,
            'rck10': rl10_avg, 'accuracy': acc_avg, 'ap': ap_avg, 'f1': f1_avg, 'macro_auc': macro_auc_avg, 'micro_auc': micro_auc_avg}

def train_windows_topo(loggers, model, meta_model, optimizer, scheduler, meta_optimizer, meta_scheduler, datasets, topo_features, **kwargs):
    num_splits = len(loggers)
    
    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    out_dir = cfg.out_dir
    print(f'Tensorboard directory: {out_dir}')
    makedirs_rm_exist(f'./{out_dir}')
    writer = SummaryWriter(f'./{out_dir}')
    with open(f'./{out_dir}/config.yaml', 'w') as f:
        cfg.dump(stream=f)
        

    model.train()

    best_param = windows_train(model, meta_model, optimizer, meta_optimizer, datasets[0], writer, topo_features)
    model.load_state_dict(best_param['best_state'])
    S_dw = best_param['best_s_dw']
    meta_model.load_state_dict(best_param['best_meta_state'])
    model.eval()
    meta_model.eval()
    perf = windows_test(datasets[1], model, meta_model, S_dw, topo_features)
    writer.add_scalars('test', perf)
    
    logging.info('Task done, results saved in {}'.format(cfg.out_dir))