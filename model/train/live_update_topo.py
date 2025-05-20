"""
The more realistic training pipeline.
"""
import copy
import datetime
import logging
import os
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score
import deepsnap
import numpy as np
import torch
from model.checkpoint import clean_ckpt
from utils.config import cfg
from model.train.live_update_train_utils import edge_index_difference, gen_negative_edges, get_keep_ratio, move_batch_to_device, report_rank_based_eval, exp_weight, sigmoid_weight, tanh_weight, adaptive_weight
from model.loss import compute_loss, compute_loss_version1
from model.utils import create_optimizer, create_scheduler
from utils.config import makedirs_rm_exist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)
weight_dict = {
    'exp': exp_weight,
    'sigmoid': sigmoid_weight,
    'tanh': tanh_weight,
    'adaptive': adaptive_weight,
}

def node_degree(edge_index, n=None, mode='in'):
    if mode == 'in':
        index = edge_index[0, :]
    elif mode == 'out':
        index = edge_index[1, :]
    else:
        index = edge_index.flatten()
    n = edge_index.max() + 1 if n is None else n
    degree = torch.zeros(n)
    ones = torch.ones(index.shape[0])
    return degree.scatter_add_(0, index, ones)

def average_state_dict(dict1: dict, dict2: dict, weight: torch.Tensor) -> dict:
    assert 0 <= weight <= 1
    out = {}
    for key in dict1.keys():
        out[key] = (1 - weight) * dict1[key] + weight * dict2[key]
    return out


def precompute_edge_degree_info(dataset):
    """Pre-computes edge_degree_existing, edge_degree_new and keep ratio
    at each snapshot. Inplace modifications.
    """
    num_nodes = dataset[0].node_feature.shape[0]
    for t in tqdm(range(len(dataset)), desc='precompute edge deg info'):
        if t == 0:
            dataset[t].node_degree_existing = torch.zeros(num_nodes)
        else:
            dataset[t].node_degree_existing \
                = dataset[t - 1].node_degree_existing \
                  + dataset[t - 1].node_degree_new

        dataset[t].node_degree_new = node_degree(dataset[t].edge_index,
                                                 n=num_nodes)

        dataset[t].keep_ratio = get_keep_ratio(
            existing=dataset[t].node_degree_existing,
            new=dataset[t].node_degree_new,
            mode=cfg.transaction.keep_ratio)
        dataset[t].keep_ratio = dataset[t].keep_ratio.unsqueeze(-1)


@torch.no_grad()
def get_task_batch(dataset: deepsnap.dataset.GraphDataset,
                   today: int, tomorrow: int,
                   prev_node_states: Optional[Dict[str, List[torch.Tensor]]]
                   ) -> deepsnap.graph.Graph:
    """
    Construct batch required for the task (today, tomorrow). As defined in
    batch's get_item method (used to get edge_label and get_label_index),
    edge_label and edge_label_index returned would be different everytime
    get_task_batch() is called.

    Moreover, copy node-memories (node_states and node_cells) to the batch.
    """
    assert today < tomorrow < len(dataset)
    # Get edges for message passing and prediction task.
    batch = dataset[today].clone()
    batch.edge_label = dataset[tomorrow].edge_label.clone()
    batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()

    # Copy previous memory to the batch.
    if prev_node_states is not None:
        for key, val in prev_node_states.items():
            copied = [x.detach().clone() for x in val]
            setattr(batch, key, copied)

    batch = move_batch_to_device(batch, cfg.device)
    return batch


@torch.no_grad()
def update_node_states(model, dataset, task: Tuple[int, int],
                       prev_node_states: Optional[
                           Dict[str, List[torch.Tensor]]]
                       ) -> Dict[str, List[torch.Tensor]]:
    """Perform the provided task and keep track of the latest node_states.

    Example: task = (t, t+1),
        the prev_node_states contains node embeddings at time (t-1).
        the model perform task (t, t+1):
            Input: (node embedding at t - 1, edges at t).
            Output: possible transactions at t+1.
        the model also generates node embeddings at t.

    after doing task (t, t+1), node_states contains information
    from snapshot t.
    """
    today, tomorrow = task
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()
    # Let the model modify batch.node_states (and batch.node_cells).
    _, _ = model(batch)
    # Collect the updated node states.
    out = dict()
    out['node_states'] = [x.detach().clone() for x in batch.node_states]
    if isinstance(batch.node_cells[0], torch.Tensor):
        out['node_cells'] = [x.detach().clone() for x in batch.node_cells]

    return out


def train_step(model, optimizer, scheduler, dataset, 
               task: Tuple[int, int],
               prev_node_states: Optional[Dict[str, torch.Tensor]]) -> dict:
    """
    After receiving ground truth from a particular task, update the model by
    performing back-propagation.
    For example, on day t, the ground truth of task (t-1, t) has been revealed,
    train the model using G[t-1] for message passing and label[t] as target.
    """
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    today, tomorrow = task
    model.train()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

    pred, true = model(batch)
    loss, pred_score = compute_loss(pred, true)
    loss.backward()
    
    optimizer.step()

    scheduler.step()
    return {'loss': loss}

def evaluate_step_grad(model, dataset, task: Tuple[int, int],
                      prev_node_states: Optional[Dict[str, List[torch.Tensor]]],
                      fast: bool = False) -> dict:
    """
    Evaluate model's performance on task = (today, tomorrow)
        where today and tomorrow are integers indexing snapshots.
    """
    today, tomorrow = task
    model.eval()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

    pred, true = model(batch)
    loss, pred_score = compute_loss_version1(pred, true)

    return loss


@torch.no_grad()
def evaluate_step(model, dataset, task: Tuple[int, int],
                  prev_node_states: Optional[Dict[str, List[torch.Tensor]]],
                  fast: bool = False) -> dict:
    """
    Evaluate model's performance on task = (today, tomorrow)
        where today and tomorrow are integers indexing snapshots.
    """
    today, tomorrow = task
    model.eval()
    batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

    pred, true = model(batch)
    loss, pred_score = compute_loss(pred, true)
    micro_auc = roc_auc_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy(), average='micro')
    macro_auc = roc_auc_score(y_true=true.detach().cpu().numpy(), y_score=pred_score.detach().cpu().numpy(), average='macro')

    pred_label = torch.zeros(len(pred_score))
    pred_label[pred_score >= 0.5] = 1.0
    accuracy = np.mean(true.detach().cpu().numpy() == pred_label.numpy())
    if fast:
        # skip MRR calculation for internal validation.
        return {'loss': loss.item(), 'micro_auc': micro_auc, 'macro_auc': macro_auc, 'accuracy': accuracy}

    mrr_batch = get_task_batch(dataset, today, tomorrow,
                               prev_node_states).clone()

    mrr, rck1, rck3, rck10 = report_rank_based_eval(
        mrr_batch, model,
        num_neg_per_node=cfg.experimental.rank_eval_multiplier)

    return {'loss': loss.item(), 'mrr': mrr, 'rck1': rck1, 'rck3': rck3,
            'rck10': rck10, 'micro_auc': micro_auc, 'macro_auc': macro_auc, 'accuracy': accuracy}


def generate_weights(meta_model, features):
    meta_model.train()
    features = features.to(torch.device(cfg.device))
    base_weights = meta_model(features)
    # 修改这里的操作方式
    # 可以选择以下激活函数:
    base_weights = torch.sigmoid(base_weights)  # 值域(0,1)
    # base_weights = torch.tanh(base_weights)     # 值域(-1,1) 
    # base_weights = torch.relu(base_weights)     # 值域(0,+∞)
    # base_weights = torch.softmax(base_weights, dim=-1)  # 值域(0,1)且和为1
    # base_weights = torch.clamp(base_weights, 0, 1)     # 硬截断到[0,1]
    # base_weights = torch.sigmoid(base_weights)  # 这里使用sigmoid作为默认选择
    # base_weights = torch.add(torch.mul(base_weights, 0.8), 0.1)
    return base_weights


def compute_parameter_smoothness(lr_tensor):
    """计算学习率的平滑性损失"""
    # 确保输入是tensor并且需要梯度
    if not lr_tensor.requires_grad:
        lr_tensor.requires_grad_(True)
        
    # 计算相邻层学习率的差异
    diff = lr_tensor[1:] - lr_tensor[:-1]
    
    # 使用L2范数计算平滑性损失
    smoothness = torch.norm(diff, p=2)
    
    # 返回标量损失
    return smoothness / lr_tensor.size(0)

def combine_parameters(current, prev, weight, mode='linear'):
    if mode == 'linear':
        return torch.add(torch.mul(current, weight), 
                        torch.mul(prev, (1-weight)))
    elif mode == 'exp':
        return torch.add(torch.mul(current, torch.exp(-weight)), 
                        torch.mul(prev, (1 - torch.exp(-weight))))
    elif mode == 'gate':
        return torch.add(torch.mul(current, torch.sigmoid(-weight)), 
                        torch.mul(prev, torch.sigmoid(weight)))
    
def generate_learning_rates(meta_model, features):
    meta_model.train()
    features = features.to(torch.device(cfg.device))
    learning_rates = meta_model(features)
    # 使用sigmoid确保学习率为正且在合理范围内
    # 使用ReLU+clamp的组合来限制学习率范围
    # ReLU确保学习率非负,clamp进一步限制上界
    # 这样可以避免sigmoid在接近0和1时梯度消失的问题
    # learning_rates = torch.sigmoid(learning_rates) * 0.2  # 将学习率限制在0到0.1之间
    learning_rates = torch.clamp(torch.relu(learning_rates), min=1e-4, max=0.01)
    return learning_rates

def train_live_update_topo(loggers, model, meta_model, optimizer, scheduler, meta_optimizer, meta_scheduler, datasets, topo_features,
                      **kwargs):

    for dataset in datasets:
        # Sometimes edge degree info is already included in dataset.
        if not hasattr(dataset[0], 'keep_ratio'):
            precompute_edge_degree_info(dataset)

    num_splits = len(loggers)  # train/val/test splits.
    # range for today in (today, tomorrow) task pairs.
    task_range = range(len(datasets[0]) - cfg.transaction.horizon)

    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')


    out_dir = cfg.out_dir
    print(f'Tensorboard directory: {out_dir}')
    makedirs_rm_exist(f'./{out_dir}')
    writer = SummaryWriter(f'./{out_dir}')
    with open(f'./{out_dir}/config.yaml', 'w') as f:
        cfg.dump(stream=f)

    prev_node_states = None  # no previous state on day 0.
    # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

    model_init = None  # for meta-learning only, a model.state_dict() object.
    auc_hist = list()
    mrr_hist = list()
    time_per_snapshot = []

    for t in tqdm(task_range, desc='snapshot', leave=True):
        start_time = datetime.datetime.now()
        for i in range(1, num_splits):
            
            if i == 1:
                perf = evaluate_step(model, datasets[i], (t, t + 1), prev_node_states)
            if i == 2:
                perf = evaluate_step(model, datasets[i], (t, t + 1), prev_node_states)
                print(perf)
                auc_hist.append(perf['micro_auc'])
                mrr_hist.append(perf['mrr'])
            writer.add_scalars('val' if i == 1 else 'test', perf, t)

        del optimizer, scheduler 
        optimizer = create_optimizer(cfg.optim.optimizer, model, cfg.optim.base_lr, cfg.optim.weight_decay)
        scheduler = create_scheduler(optimizer)
        if t > 0:
            prev_params = copy.deepcopy(best_model['state'])
        best_model = {'val_loss': np.inf, 'train_epoch': 0, 'state': None}
        best_model_unchanged = 0
        tol = cfg.train.internal_validation_tolerance
        for i in tqdm(range(cfg.optim.max_epoch + 1), desc='live update',
                    leave=True):
            # Start with the un-trained model (i = 0), evaluate the model.
            internal_val_perf = evaluate_step(model, datasets[1],
                                            (t, t + 1),
                                            prev_node_states, fast=True)
            val_loss = internal_val_perf['loss']

            if val_loss < best_model['val_loss']:
                # replace the best model with the current model.
                best_model = {'val_loss': val_loss, 'train_epoch': i,
                            'state': copy.deepcopy(model.state_dict())}
                best_model_unchanged = 0
            else:
                # the current best model has dominated for these epochs.
                best_model_unchanged += 1

            # if (i >= 2 * tol) and (best_model_unchanged >= tol):
            if best_model_unchanged >= tol:
                # If the best model has not been updated for a while, stop.
                break
            else:
                # Otherwise, keep training.
                train_perf = train_step(model, optimizer, scheduler,
                                        datasets[0], (t, t + 1),
                                        prev_node_states)
                if cfg.roland.is_meta and (model_init is not None):
                    writer.add_scalars('train', train_perf, t)

        writer.add_scalar('internal_best_val', best_model['val_loss'], t)
        writer.add_scalar('best epoch', best_model['train_epoch'], t)

        # (3) Actually perform the update on training set to get node_states
        # contains information up to time t.
        # Use the best model selected from intra-snapshot cross-validation.
        model.load_state_dict(best_model['state'])
        mask = torch.bernoulli(torch.tensor(1. - cfg.topo.drop_rate))
        if t > 0 and mask.item():
            # 初始化最佳meta损失和状态
            best_meta_loss = float('inf')
            best_meta_state = None
            
            # 进行多次meta优化循环
            for meta_iter in range(cfg.optim.meta_loop):
                
                # 1. 计算多个时间片的第一个损失和
                with torch.set_grad_enabled(True):
                    loss_1 = 0
                    for prev_t in range(max(0, t-cfg.train.memory_steps), t):
                        # 计算时间衰减权重
                        time_diff = t - prev_t
                        decay_weight = exp_weight(time_diff, alpha0=1.0, gamma=0.5, delta=0)
                        loss_1 += decay_weight * evaluate_step_grad(model, datasets[1], (prev_t, prev_t + 1),
                                                prev_node_states, fast=True)
                    loss_1_value = loss_1.clone().detach()  # 保存loss_1的值
                
                # 计算参数的梯度
                loss_1.backward()
                
                # 保存梯度和对应的参数名
                gradients = {}
                param_names_with_grad = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone()
                        param_names_with_grad.append(name)
                
                # 清除梯度
                model.zero_grad()
                
                # 生成学习率并更新参数
                learning_rates = generate_learning_rates(meta_model, (topo_features[t] - topo_features[t-1]))
                
                current_params = copy.deepcopy(model.state_dict())
                # 只更新有梯度的参数
                for i, name in enumerate(param_names_with_grad):
                    if i < len(learning_rates):  # 确保不会超出learning_rates的范围
                        # 更新参数
                        current_params[name] = current_params[name] - learning_rates[i] * gradients[name]
                
                model.load_state_dict(current_params)
                
                # 3. 计算多个时间片的第二个损失和
                with torch.set_grad_enabled(True):
                    loss_2 = 0
                    for prev_t in range(max(0, t-cfg.train.memory_steps), t):
                        # 计算时间衰减权重
                        time_diff = t - prev_t
                        decay_weight = exp_weight(time_diff, alpha0=1.0, gamma=0.5, delta=0)
                        loss_2 += decay_weight * evaluate_step_grad(model, datasets[1], (prev_t, prev_t + 1),
                                                prev_node_states, fast=True)
                
                # 4. 计算meta损失，使用保存的loss_1值
                margin = 0.1
                diff_loss = torch.sub(loss_2, loss_1_value)
                meta_val_loss = torch.relu(diff_loss + margin)
                
                # 计算参数平滑度损失
                smoothness_loss = compute_parameter_smoothness(learning_rates)
                meta_val_loss = meta_val_loss + 0.1 * smoothness_loss.detach()  # 使用detach()避免重复计算梯度
                
                # 记录最佳结果
                if meta_val_loss.item() < best_meta_loss:
                    best_meta_loss = meta_val_loss.item()
                    best_meta_state = copy.deepcopy(model.state_dict())
                
                # 5. 优化元模型
                meta_optimizer.zero_grad()
                meta_val_loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)
                meta_optimizer.step()
                meta_scheduler.step()
                
                writer.add_scalar('meta/val_loss', meta_val_loss.item(), t)
                writer.add_scalar('meta/avg_weight', learning_rates.mean().item(), t)
            
            # 加载最佳的模型状态
            if best_meta_state is not None:
                model.load_state_dict(best_meta_state)
        
        prev_node_states = update_node_states(model, datasets[0], (t, t + 1),
                                              prev_node_states)
        
        # 计算并记录当前时间片的耗时
        end_time = datetime.datetime.now()
        time_delta = (end_time - start_time).total_seconds()
        time_per_snapshot.append(time_delta)
        
    # 计算并打印平均耗时
    avg_time = sum(time_per_snapshot) / len(time_per_snapshot)
    print(f"Average time per snapshot: {avg_time:.2f} seconds")

    writer.close()

    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))

    # # for debugging purpose.
    # import matplotlib.pyplot as plt
    # print('AUC = ', np.mean(auc_hist))
    # print('MRR = ', np.mean(mrr_hist))
    # plt.plot(auc_hist, label='AUC')
    # plt.plot(mrr_hist, label='MRR')
    # plt.legend()
    # plt.savefig('temp_out.png')