#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchmetrics
from utils.config import cfg
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score

def compute_loss_meta(pred, y):
    L = nn.BCELoss()
    pred = pred.float()
    y = y.to(pred)
    loss = L(pred, y)

    return loss


def prediction(pred_score, true_l):
    # Acc_torch = torchmetrics.Accuracy(task='binary').to(pred_score)
    # Macro_Auc_torch = torchmetrics.AUROC(task='binary', average='macro').to(pred_score)
    # Micro_Auc_torch = torchmetrics.AUROC(task='binary', average='micro').to(pred_score)
    # Ap_torch = torchmetrics.AveragePrecision(task='binary').to(pred_score)
    # F1_torch = torchmetrics.F1Score(task='binary', average='macro').to(pred_score)
    #
    # acc_torch = Acc_torch(pred_score, true_l.to(pred_score))
    # macro_auc_torch = Macro_Auc_torch(pred_score, true_l.to(pred_score))
    # micro_auc_torch = Micro_Auc_torch(pred_score, true_l.to(pred_score))
    # ap_torch = Ap_torch(pred_score, true_l.to(pred_score))
    # f1_torch = F1_torch(pred_score, true_l.to(pred_score))

    pred = pred_score.clone()
    pred = torch.where(pred > 0.5, 1, 0)
    pred = pred.detach().cpu().numpy()
    pred_score = pred_score.detach().cpu().numpy()

    # true = np.ones_like(pred)
    true = true_l
    true = true.cpu().numpy()
    acc = accuracy_score(true, pred)
    ap = average_precision_score(true, pred_score)
    f1 = f1_score(true, pred, average='macro')
    macro_auc = roc_auc_score(true, pred_score, average='macro')
    micro_auc = roc_auc_score(true, pred_score, average='micro')

    # print(acc, ap, f1, macro_auc, micro_auc)
    # print(acc_torch, ap_torch, f1_torch, macro_auc_torch, micro_auc_torch)
    return acc, ap, f1, macro_auc, micro_auc
    # return acc_torch, ap_torch, f1_torch, macro_auc_torch, micro_auc_torch

def compute_loss(pred, true):
    '''

    :param pred: unnormalized prediction
    :param true: label
    :return: loss, normalized prediction score
    '''
    bce_loss = nn.BCEWithLogitsLoss(size_average=cfg.model.size_average)
    mse_loss = nn.MSELoss(size_average=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    # if multi task binary classification, treat as flatten binary
    if true.ndim > 1 and cfg.model.loss_fun == 'cross_entropy':
        pred, true = torch.flatten(pred), torch.flatten(true)
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        # binary
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    else:
        raise ValueError('Loss func {} not supported'.
                         format(cfg.model.loss_fun))


def compute_loss_version1(pred, true):
    '''
    :param pred: unnormalized prediction
    :param true: label
    :return: loss, normalized prediction score
    '''
    bce_loss = nn.BCEWithLogitsLoss(size_average=cfg.model.size_average)
    mse_loss = nn.MSELoss(size_average=cfg.model.size_average)

    # 创建新的张量而不是修改原始张量
    pred_processed = pred.clone()
    true_processed = true.clone()

    # 处理多任务二分类情况
    if true_processed.ndim > 1 and cfg.model.loss_fun == 'cross_entropy':
        pred_processed = torch.flatten(pred_processed)
        true_processed = torch.flatten(true_processed)
    
    # 使用clone()避免原地操作
    if pred_processed.ndim > 1:
        pred_processed = pred_processed.squeeze(-1).clone()
    if true_processed.ndim > 1:
        true_processed = true_processed.squeeze(-1).clone()

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred_processed.ndim > 1:
            pred_softmax = F.log_softmax(pred_processed, dim=-1)
            return F.nll_loss(pred_softmax, true_processed), pred_softmax
        # binary
        else:
            true_float = true_processed.float()
            pred_sigmoid = torch.sigmoid(pred_processed)
            return bce_loss(pred_processed, true_float), pred_sigmoid
    elif cfg.model.loss_fun == 'mse':
        true_float = true_processed.float()
        return mse_loss(pred_processed, true_float), pred_processed
    else:
        raise ValueError('Loss func {} not supported'.
                         format(cfg.model.loss_fun))