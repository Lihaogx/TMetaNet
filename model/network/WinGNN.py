#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from utils.config import cfg
from dgl.nn.pytorch.conv import GraphConv
from torch_scatter import scatter_add
from copy import deepcopy
from model.layer.windows_layer import Windows_GCNLayer

layer_dict = {
    'windows_gcn': Windows_GCNLayer
}


class WinGNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(WinGNN, self).__init__()
        self.num_layers = cfg.gnn.layers_mp
        self.dropout = cfg.gnn.dropout
        self.hidden_dim = cfg.gnn.hidden_dim
        for i in range(self.num_layers):
            d_in = dim_in if i == 0 else cfg.gnn.dim_out
            pos_isn = True if i == 0 else False
            layer = layer_dict[cfg.gnn.layer_type](dim_in=d_in, dim_out=cfg.gnn.dim_out, pos_isn=pos_isn)
            self.add_module('gnn{}'.format(i), layer)


        self.weight1 = nn.Parameter(torch.ones(size=(self.hidden_dim, cfg.gnn.dim_out)))
        self.weight2 = nn.Parameter(torch.ones(size=(dim_out, self.hidden_dim)))

        if cfg.model.edge_decoding == 'dot':
            self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
        elif cfg.model.edge_decoding == 'cosine_similarity':
            self.decode_module = nn.CosineSimilarity(dim=-1)
        else:
            raise ValueError('Unknown edge decoding {}.'.format(
                cfg.model.edge_decoding))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight1, gain=gain)
        nn.init.xavier_normal_(self.weight2, gain=gain)

    def forward(self, g, x, fast_weights=None):
        # 确保输入在同一设备上
        device = x.device
        if fast_weights is not None:
            fast_weights = [w.to(device) for w in fast_weights]
        
        count = 0
        for i in range(self.num_layers):
            gnn = getattr(self, f'gnn{i}')
            gcn_start = 2 + i * 4
            gcn_end = gcn_start + 4
            gcn_fast_weights = fast_weights[gcn_start:gcn_end] if fast_weights else None
            x = gnn(g, x, gcn_fast_weights)

            count += 1

        if fast_weights:
            weight1 = fast_weights[0]
            weight2 = fast_weights[1]
        else:
            weight1 = self.weight1
            weight2 = self.weight2

        x = F.normalize(x)
        g.node_embedding = x

        pred = F.dropout(x, self.dropout)
        pred = F.relu(F.linear(pred, weight1))
        pred = F.dropout(pred, self.dropout)
        pred = torch.sigmoid(F.linear(pred, weight2)) 

        node_feat = pred[g.edge_label_index]
        nodes_first = node_feat[0]
        nodes_second = node_feat[1]

        pred = self.decode_module(nodes_first, nodes_second)

        return pred

