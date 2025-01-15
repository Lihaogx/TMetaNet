#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
from typing import Optional, List, Type
import dgl
import deepsnap
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim
from utils.config import cfg
from torch_scatter import scatter_max, scatter_mean, scatter_min
from model.loss import prediction


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = cfg.optim.momentum
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer

def create_scheduler(optimizer):
    # Try to load customized scheduler
    if cfg.optim.scheduler == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.optim.max_epoch + 1)
    elif cfg.optim.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.optim.steps,
                                                   gamma=cfg.optim.lr_decay)
    elif cfg.optim.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=cfg.optim.max_epoch)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            cfg.optim.scheduler))
    return scheduler


def create_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "softmax":
        return nn.Softmax()

    raise RuntimeError("activation error, not {}".format(activation))

