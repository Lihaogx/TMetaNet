import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from model.layer.act_layer import act_dict
from utils.config import cfg


class ResidualEdgeConvLayer(MessagePassing):
    r"""General GNN layer, with arbitrary edge features.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(ResidualEdgeConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction

        if self.msg_direction == 'single':
            self.linear_msg = nn.Linear(in_channels + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        elif self.msg_direction == 'both':
            self.linear_msg = nn.Linear(in_channels * 2 + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        else:
            raise ValueError

        if cfg.gnn.skip_connection == 'affine':
            self.linear_skip = nn.Linear(in_channels, out_channels, bias=True)
        elif cfg.gnn.skip_connection == 'identity':
            assert self.in_channels == self.out_channels

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        if cfg.gnn.skip_connection == 'affine':
            skip_x = self.linear_skip(x)
        elif cfg.gnn.skip_connection == 'identity':
            skip_x = x
        else:
            skip_x = 0
        return self.propagate(edge_index, x=x, norm=norm,
                              edge_feature=edge_feature) + skip_x

    def message(self, x_i, x_j, norm, edge_feature):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        elif self.msg_direction == 'single':
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        else:
            raise ValueError
        x_j = self.linear_msg(x_j)
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ResidualEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(ResidualEdgeConv, self).__init__()
        self.model = ResidualEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch

layer_dict = {
    'residual_edge_conv': ResidualEdgeConv
}

class RolandLayer(nn.Module):
    """
    The most general wrapper for graph recurrent layer, users can customize
        (1): the GNN block for message passing.
        (2): the update block takes {previous embedding, new node feature} and
            returns new node embedding.
    """
    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, id=0, **kwargs):
        super(RolandLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer_id = id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer = layer_dict[cfg.gnn.layer_type](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
            
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.embedding_updater = GRUUpdater(self.dim_in, self.dim_out, self.layer_id)

    def _init_hidden_state(self, batch):
        # Initialize hidden states of all nodes to zero.
        if not isinstance(batch.node_states[self.layer_id], torch.Tensor):
            batch.node_states[self.layer_id] = torch.zeros(
                batch.node_feature.shape[0], self.dim_out).to(
                batch.node_feature.device)

    def forward(self, batch):
        # Message passing.
        batch = self.layer(batch)
        batch.node_feature = self.post_layer(batch.node_feature)
        if self.has_l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)

        self._init_hidden_state(batch)
        node_states_new = self.embedding_updater(batch)
        batch.node_states[self.layer_id] = node_states_new
        batch.node_feature = batch.node_states[self.layer_id]
        return batch

class GRUUpdater(nn.Module):
    """
    Node embedding update block using standard GRU and variations of it.
    """
    def __init__(self, dim_in, dim_out, layer_id):
        # dim_in (dim of X): dimension of input node_feature.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super(GRUUpdater, self).__init__()
        self.layer_id = layer_id
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())
    
    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        # if cfg.gnn.embed_update_method == 'masked_gru':
        #     # Update for active nodes only, use output from GRU.
        #     keep_mask = (batch.node_degree_new == 0)
        #     H_out = H_gru
        #     # Reset inactive nodes' embedding.
        #     H_out[keep_mask, :] = H_prev[keep_mask, :]
        # elif cfg.gnn.embed_update_method == 'moving_average_gru':
        #     # Only update for active nodes, using moving average with output from GRU.
        #     H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
        # elif cfg.gnn.embed_update_method == 'gru':
        #     # Update all nodes' embedding using output from GRU.
        #     H_out = H_gru
        H_out = H_gru
        return H_out