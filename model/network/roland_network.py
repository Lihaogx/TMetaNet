import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg
from model.layer.act_layer import act_dict
from model.layer.feature_processors import Preprocess, Postprocess
from model.init import init_weights
from model.layer.roland_layer import RolandLayer
########### Layer ############
# Methods to construct layers.

class GNNStackStage(nn.Module):
    r"""Simple Stage that stacks GNN layers"""

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = RolandLayer(cfg.gnn.layer_type, dim_in, dim_out,
                                              has_act=True, id=i)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch
        


########### Model: start + stage + head ############

class Roland(nn.Module):
    r"""The General GNN model"""

    def __init__(self, dim_in, dim_out, **kwargs):
        r"""Initializes the GNN model.

        Args:
            dim_in, dim_out: dimensions of in and out channels.
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(Roland, self).__init__()
        GNNStage = GNNStackStage
        self.preprocess = Preprocess(dim_in, cfg.gnn.dim_inner)
        if cfg.gnn.layers_mp >= 1:
            self.mp = GNNStage(dim_in=cfg.gnn.dim_inner,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            d_in = self.mp.dim_out
        self.postprocess = Postprocess(dim_in=d_in, dim_out=dim_out)

        self.apply(init_weights)
            
    def forward(self, batch):
        batch = self.preprocess(batch)
        batch = self.mp(batch)
        batch = self.postprocess(batch)
        return batch