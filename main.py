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
import logging
from datetime import datetime
from tqdm import tqdm
from model.network import WinGNN
from utils.config import cfg, update_out_dir, assert_cfg, dump_cfg
from utils.logger import setup_printing, create_logger
from deepsnap.graph import Graph
from model.loader import create_dataset, create_model
from model.utils import create_optimizer, create_scheduler
from utils.devices import auto_select_device
from utils.agg_runs import agg_runs
from utils.config import get_parent_dir
from deepsnap.dataset import GraphDataset
from model.train.live_update import train_live_update
from model.train.windows import train_windows
from model.train.windows_dgl import train_windows_dgl
import yaml
import warnings
warnings.filterwarnings("ignore")
train_dict = {
    'live_update': train_live_update,
    'windows': train_windows,
    'windows_dgl': train_windows_dgl
}

def dataset_cfg_setup_live_update(name: str):
    """
    Setup required fields in cfg for the given dataset.
    """

    if name in ['bitcoin-otc', 'bitcoin-alpha']:
        cfg.dataset.edge_dim = 2
        cfg.transaction.snapshot_freq = 'W'
        
    elif name in ['uci-msg']:
        cfg.dataset.edge_dim = 1
        cfg.transaction.snapshot_freq = 'W'
        
    elif name in ['reddit-body', 'reddit-title']:
        cfg.dataset.edge_dim = 88
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['as']:
        cfg.dataset.edge_dim = 1
        cfg.transaction.snapshot_freq = 'D'

    else:
        raise ValueError(f'No default config for dataset {name}.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    cfg.merge_from_file(args.config)

    torch.set_num_threads(cfg.num_threads)
    out_dir_parent = cfg.out_dir
    cfg.seed = random.randint(1, 10000)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    update_out_dir(out_dir_parent, args.config)
    dump_cfg(cfg)
    setup_printing()
    auto_select_device(cfg)
    dataset_cfg_setup_live_update(cfg.dataset.name)
    
    datasets = create_dataset()
    model = create_model(datasets)
    meters = create_logger(datasets, datasets)
    
    optimizer = create_optimizer(cfg.optim.optimizer, model, cfg.optim.base_lr, cfg.optim.weight_decay)
    scheduler = create_scheduler(optimizer)
    
    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = sum([p.numel() for p in model.parameters()])
    logging.info('Num parameters: {}'.format(cfg.params))
    
    
    # for dataset, name in zip(datasets, ('train', 'validation', 'test')):
    #     print(f'{name} set: {len(dataset)} graphs.')
    #     all_edge_time = torch.cat([g.edge_time for g in dataset])
    #     start = int(torch.min(all_edge_time))
    #     start = datetime.fromtimestamp(start)
    #     end = int(torch.max(all_edge_time))
    #     end = datetime.fromtimestamp(end)
    #     print(f'\tRange: {start} - {end}')
    
    train_dict[cfg.train.mode](meters, model, optimizer, scheduler, datasets)