from utils.config import cfg
from typing import Optional, List, Type
import deepsnap
import os
import numpy as np
from scipy.sparse import coo_matrix
from deepsnap.graph import Graph
import dgl
import torch
from tqdm import tqdm
from deepsnap.dataset import GraphDataset
from model.network.roland_network import Roland
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from model.roland_data_loader.roland_btc import load_btc_dataset
from model.network.WinGNN import WinGNN
import math
network_dict = {
    'roland': Roland,
    'wingnn': WinGNN
}
loader_dict={
    'bitcoin-alpha': load_btc_dataset,
}
def read_npz(path):
    filesname = os.listdir(path)
    npz = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('.')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        npz.append(np.load(path+filename))

    return npz


def create_dataset():
    path = "dataset/" + cfg.dataset.name
    # 
    if cfg.train.mode == 'windows_dgl':
        path_ei = path + '/' + 'edge_index/'
        path_nf = path + '/' + 'node_feature/'
        path_ef = path + '/' + 'edge_feature/'
        path_et = path + '/' + 'edge_time/'
        
        edge_index = read_npz(path_ei)
        e_feat = read_npz(path_ef)
        n_feat = read_npz(path_nf)
        e_time = read_npz(path_et)

        nodes_num = n_feat[0].shape[0]

        graphs = []
        for e_i in edge_index:
            row = e_i[0]
            col = e_i[1]
            ts = [1] * len(row)
            sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
            graphs.append(sub_g)
        
        n_dim = n_feat[0].shape[1]
        cfg.n_dim = n_dim
        n_node = n_feat[0].shape[0]
        cfg.n_node = n_node
        graph_l = []
        # Data set processing
        for idx, graph in tqdm(enumerate(graphs)):
            
            # DGL格式处理
            graph_d = dgl.from_scipy(graph)

            graph_d.edge_feature = torch.Tensor(e_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])

            if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
                n_feat_t = graph_l[idx - 1].node_feature
                graph_d.node_feature = torch.Tensor(n_feat_t)
            else:
                graph_d.node_feature = torch.Tensor(n_feat[idx])

            graph_d = dgl.remove_self_loop(graph_d)
            graph_d = dgl.add_self_loop(graph_d)

            edges = graph_d.edges()
            row = edges[0].numpy()
            col = edges[1].numpy()
            # Negative sample sampling 1:1
            n_e = graph_d.num_edges() - graph_d.num_nodes()
            # Edge label
            y_pos = np.ones(shape=(n_e,))
            y_neg = np.zeros(shape=(n_e,))
            y = list(y_pos) + list(y_neg)

            edge_label_index = list()
            edge_label_index.append(row.tolist()[:n_e])
            edge_label_index.append(col.tolist()[:n_e])

            graph_d.edge_label = torch.Tensor(y)
            graph_d.edge_label_index = torch.LongTensor(edge_label_index)
            graph_l.append(graph_d)
        for idx, graph in tqdm(enumerate(graphs)):
            graph = Graph(
                node_feature=graph_l[idx].node_feature,
                edge_feature=graph_l[idx].edge_feature,
                edge_index=graph_l[idx].edge_label_index,
                edge_time=graph_l[idx].edge_time,
                directed=True
            )

            dataset = GraphDataset(graph,
                                   task='link_pred',
                                   edge_negative_sampling_ratio=1.0,
                                   minimum_node_per_graph=5)
            edge_labe_index = dataset.graphs[0].edge_label_index
            graph_l[idx].edge_label_index = torch.LongTensor(edge_labe_index)
        
    # elif cfg.model.type == 'roland':
    #     graph_l = loader_dict[cfg.dataset.name](path, cfg.dataset.name)
        
    if cfg.dataset.task_splitting == 'temporal':

        n = math.ceil(len(graph_l) * cfg.dataset.split[0])
        dataset_train = graph_l[:n]
        dataset_test = graph_l[n:]
        datasets = [dataset_train, dataset_test]
        # datasets = GraphDataset(graphs=graph_l,
        #                         task='link_pred',
        #                         edge_train_mode='all',
        #                         minimum_node_per_graph=5)
        # datasets.split(
        #         transductive=False,
        #         split_ratio=cfg.dataset.split,
        #         shuffle=False)
    elif cfg.dataset.task_splitting == 'within':
        graph_l = loader_dict[cfg.dataset.name](path, cfg.dataset.name)
        dataset = GraphDataset(graphs=graph_l,
                            task='link_pred',
                            edge_train_mode='all',
                            minimum_node_per_graph=5)
        datasets = dataset.split(
                transductive=True,
                split_ratio=cfg.dataset.split,
                shuffle=True)
    else:
        raise ValueError(f"Invalid task splitting method: {cfg.task_splitting}")
    return datasets

def create_model(
    datasets: Optional[List[deepsnap.dataset.GraphDataset]] = None,
    to_device: bool = True,
    dim_in: Optional[int] = None,
    dim_out: Optional[int] = None
) -> Type[torch.nn.Module]:
    r"""Constructs the pytorch-geometric model.

    Args:
        datasets: A list of deepsnap.dataset.GraphDataset objects.
        to_device: A bool indicating whether to move the constructed model to the device in configuration.
        dim_in: An integer indicating the input dimension of the model.
            If not provided (None), infer from datasets and use `num_node_features`
        dim_out: An integer indicating the output dimension of the model
            If not provided (None), infer from datasets and use `num_node_features`
    Returns:
        The constructed pytorch model.
    """
    # FIXME: num_node_features/num_labels not working properly for HeteroGraph.
    if cfg.train.mode == 'windows_dgl':
        dim_in = 1
        dim_out = 2
    else:
        dim_in = datasets[0].num_node_features if dim_in is None else dim_in
        dim_out = datasets[0].num_labels if dim_out is None else dim_out
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        # binary classification, output dim = 1
        dim_out = 1
    model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)
    if to_device:
        model.to(torch.device(cfg.device))
    return model