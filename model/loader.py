from utils.config import cfg
from typing import Optional, List, Type
import deepsnap
import os
import time
from tqdm import tqdm
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
from model.roland_data_loader.roland_reddit_hyperlink import load_reddit_dataset
from model.network.WinGNN import WinGNN
from topo_utils.homology_utils import compute_topo_diagram, compute_persistence_image
import math
from topo_utils.distance import compute_wasserstein_distance, compute_bottleneck_distance, compute_heat_distance
from model.layer.cnn_layer import MultiChannelCNN, FusionCNN, AttentionFusionCNN, ResidualFusionCNN, DilatedResidualFusionCNN, ImprovedFusionCNN
from model.roland_data_loader.roland_ucimsg import load_uci_dataset
from model.roland_data_loader.roland_eth import load_eth_dataset
distance_dict = {
    'wasserstein': compute_wasserstein_distance,
    'bottleneck': compute_bottleneck_distance,
    'heat': compute_heat_distance
}
network_dict = {
    'roland': Roland,
    'wingnn': WinGNN
}
loader_dict={
    'bitcoin-alpha': load_btc_dataset,
    'bitcoin-otc': load_btc_dataset,
    'reddit-body': load_reddit_dataset,
    'reddit-title': load_reddit_dataset,
    'uci-msg': load_uci_dataset,
    'ethereum': load_eth_dataset,
}
meta_model_dict={
    'MultiChannel': MultiChannelCNN,
    'Fusion': FusionCNN,
    'Attention': AttentionFusionCNN,
    'Residual': ResidualFusionCNN,
    'DilatedResidual': DilatedResidualFusionCNN,
    'ImprovedFusion': ImprovedFusionCNN,
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
    path = "/home/lh/Dowzag_2.0/dataset/" + cfg.dataset.name
    # 
    if 'windows' in cfg.train.mode:
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
            n = math.ceil(len(graph_l) * cfg.dataset.split[0])
            dataset_train = graph_l[:n]
            dataset_test = graph_l[n:]
            datasets = [dataset_train, dataset_test]
    elif 'live_update' in cfg.train.mode:
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
        raise ValueError(f"Invalid train mode: {cfg.train.mode}")
    return datasets


def load_topo_dataset(datasets):
    path = "/home/lh/Dowzag_2.0/dataset/" + cfg.dataset.name + "/topo_feature/"
    window_size = cfg.topo.window_size
    remove_edge = cfg.topo.remove_edge
    is_directed = cfg.topo.is_directed
    topo_diagrams = []
    topo_features_list = []
    for filtration in cfg.topo.filtration:
        epsilon, delta = filtration
            # 生成文件名
        filename = f"topo_diagram_w{window_size}_e{epsilon}_d{delta}_r{remove_edge}_d{is_directed}.pt"
        filepath = os.path.join(path, filename)
        
        # 检查文件是否存在
        if os.path.exists(filepath):
            print(f"Loading cached topo diagram from {filepath}")
            topo_diagram = torch.load(filepath)
            topo_diagrams.append(topo_diagram)
        else:
            print(f"Computing topo diagram for {filename}")
            if cfg.model.type == 'roland':
                edge_sequences = []
                for i in range(len(datasets[0])):
                    # 收集所有数据集中第i个图的边
                    edge_indices = []
                    edge_labels = []
                    for dataset in datasets:
                        edge_indices.append(dataset[i].edge_label_index)
                        edge_labels.append(dataset[i].edge_label)
                    # 分别处理每个数据集的边
                    existing_edges_list = []
                    for edge_idx, edge_lab in zip(edge_indices, edge_labels):
                        # 只保留存在的边(label=1)
                        existing_edges = edge_idx[:, edge_lab==1]
                        existing_edges_list.append(existing_edges)
                    # 将每个数据集的边合并
                    existing_edges = torch.cat(existing_edges_list, dim=1)
                    edge_sequences.append(existing_edges)
            else:
                raise ValueError("Please use Roland-type datasets for computing topological features")
            topo_diagram = compute_topo_diagram(edge_sequences, window_size, epsilon, delta, remove_edge, is_directed)
            torch.save(topo_diagram, filepath)
            topo_diagrams.append(topo_diagram)
        # 计算0维和1维的persistence image
        diagram_features = [
            compute_persistence_image(dim, topo_diagram, [cfg.topo.resolution]*2, window_size, cfg.topo.bandwidth, cfg.topo.power)
            for dim in [0,1]
        ]
        # 转换为tensor
        topo_features = [torch.tensor(f, dtype=torch.float) for f in diagram_features]
        topo_features = torch.stack(topo_features, dim=1)
        
        topo_features_list.append(topo_features)
    topo_features = torch.cat(topo_features_list, dim=1)
    distance = []
    for i in tqdm(range(len(topo_diagrams[0]) - 1)):
        dist = 0
        for j in range(len(topo_diagrams)):
            dist += distance_dict[cfg.topo.distance](topo_diagrams[j][i], topo_diagrams[j][i+1])
        distance.append(dist)
    # 对distance进行归一化处理
    distance = torch.tensor(distance)
    if len(distance) > 0:  # 确保distance不为空
        distance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-8)  # 添加小量避免除零
    distance = distance.tolist()
    return topo_diagrams, topo_features, distance
    
def create_meta_model(dim_out=1, to_device=True):
    meta_model = meta_model_dict[cfg.topo.meta_type](dim_out)
    if to_device:
        meta_model.to(torch.device(cfg.device))
    return meta_model

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
    dim_in = cfg.dataset.node_dim
    dim_out = 1
    # if cfg.model.type == 'roland':
    #     dim_in = datasets[0].num_node_features if dim_in is None else dim_in
    #     dim_out = datasets[0].num_labels if dim_out is None else dim_out
    # else:
    #     dim_in = cfg.dataset.node_dim
    #     dim_out = cfg.dataset.edge_dim
    # if 'classification' in cfg.dataset.task_type and dim_out == 2:
    #     # binary classification, output dim = 1
    #     dim_out = 1
    model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)
    if to_device:
        model.to(torch.device(cfg.device))
    return model