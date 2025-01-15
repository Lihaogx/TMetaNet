import torch
import torch.nn as nn
import torch.nn.functional as F
import deepsnap

from utils.config import cfg
from model.layer.act_layer import act_dict



class Preprocess(nn.Module):
    def __init__(self, dim_in, dim_out, has_act=True, has_bn=True):
        super(Preprocess, self).__init__()
        edge_feature_layer_wrapper = []
        if cfg.dataset.edge_encoder:
            expected_dim = cfg.transaction.feature_amount_dim + cfg.transaction.feature_time_dim
            edge_feature_layer_wrapper.append(nn.Linear(cfg.dataset.edge_dim, expected_dim))
            cfg.dataset.edge_dim = expected_dim
            if cfg.dataset.edge_encoder_bn:
                edge_feature_layer_wrapper.append(nn.BatchNorm1d(expected_dim, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        self.edge_feature_layer = nn.Sequential(*edge_feature_layer_wrapper)
                
        has_bn = has_bn and cfg.gnn.batchnorm
        node_feature_layer_wrapper = []
        node_feature_layer_wrapper.append(nn.Linear(dim_in, dim_out, bias=not has_bn))
        if has_bn:
            node_feature_layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            node_feature_layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            node_feature_layer_wrapper.append(act_dict[cfg.gnn.act])
        self.node_feature_layer = nn.Sequential(*node_feature_layer_wrapper)
        
    def forward(self, batch):
        batch.edge_feature = self.edge_feature_layer(batch.edge_feature)
        batch.node_feature = self.node_feature_layer(batch.node_feature)
        return batch
    

class Postprocess(nn.Module):
    r"""The GNN head module for edge prediction tasks. This module takes a (batch of) graphs and
    outputs ...
    """

    def __init__(self, dim_in: int, dim_out: int):
        ''' Head of Edge and link prediction models.

        Args:
            dim_out: output dimension. For binary prediction, dim_out=1.
        '''
        super(Postprocess, self).__init__()
        layers = []
        layers.append(nn.Linear(dim_in * 2, dim_in * 2, bias=True))
        layers.append(nn.BatchNorm1d(dim_in * 2, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        layers.append(nn.Dropout(p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        layers.append(act_dict[cfg.gnn.act])
        layers.append(nn.Linear(dim_in * 2, dim_out, bias=True))
        self.layer_post_mp = nn.Sequential(*layers)
        
        self.decode_module = lambda v1, v2: self.layer_post_mp(torch.cat((v1, v2), dim=-1))

    def _apply_index(self, batch):
        return batch.node_feature[batch.edge_label_index], batch.edge_label
    
    def forward(self, batch):
        pred, label = self._apply_index(batch)
        if hasattr(batch, 'device'):
            # TODO: optionally move the head-prediction to cpu to allow for
            #  higher throughput (e.g., for MRR computations).
            raise NotImplementedError
        nodes_first = pred[0]  # node features of the source node of each edge.
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)
        return pred, label