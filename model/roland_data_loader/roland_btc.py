"""
Data loader for bitcoin datasets.
Mar. 27, 2021
"""
import os
import types
from typing import List, Union

import deepsnap
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


cfg_dict = {
    "transaction": {
        "snapshot": True,
        "snapshot_freq": "W",
    },
    "train": {
        "mode": "live_update"
    },
    "gnn": {
        "layers_mp": 2,
        "noise": 0.0
    },
    "dataset": {
        "split_method": "default"
    }
}

def dict_to_namespace(d):
    namespace = types.SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace
cfg = dict_to_namespace(cfg_dict)

def load_single_dataset(dataset_dir: str) -> Graph:
    df_trans = pd.read_csv(dataset_dir, sep=',', header=None, index_col=None)
    df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']
    # NOTE: 'SOURCE' and 'TARGET' are not consecutive.
    num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

    # bitcoin OTC contains decimal numbers, round them.
    df_trans['TIME'] = df_trans['TIME'].astype(np.int64).astype(np.float64)
    assert not np.any(pd.isna(df_trans).values)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIME'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['RATING', 'TimestampScaled']].values)  # (E, edge_dim)
    # SOURCE and TARGET IDs are already encoded in the csv file.
    # edge_index = torch.Tensor(
    #     df_trans[['SOURCE', 'TARGET']].values.transpose()).long()  # (2, E)

    node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    enc = OrdinalEncoder(categories=[node_indices, node_indices])
    raw_edges = df_trans[['SOURCE', 'TARGET']].values
    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)

    # num_nodes = torch.max(edge_index) + 1
    # Use dummy node features.
    node_feature = torch.ones(num_nodes, 1).float()

    edge_time = torch.FloatTensor(df_trans['TIME'].values)

    if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
        edge_feature = torch.cat((edge_feature, edge_feature.clone()), dim=0)
        reversed_idx = torch.stack([edge_index[1], edge_index[0]]).clone()
        edge_index = torch.cat((edge_index, reversed_idx), dim=1)
        edge_time = torch.cat((edge_time, edge_time.clone()))

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph


def make_graph_snapshot(g_all: Graph, snapshot_freq: str) -> List[Graph]:
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()

    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time)))

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'  # month of year.
                }

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()

    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)

    snapshot_list.sort(key=lambda x: torch.min(x.edge_time))

    return snapshot_list


def split_by_seconds(g_all, freq_sec: int):
    # Split the entire graph into snapshots.
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        period_members = (split_criterion == t)
        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)
    return snapshot_list


def add_structural_noise(graph, noise_ratio=0.1):
    """为图添加结构噪声
    Args:
        graph: deepsnap.Graph对象
        noise_ratio: 要添加/删除的边的比例
    """
    num_edges = graph.edge_index.size(1)
    num_noise_edges = int(num_edges * noise_ratio)
    
    # 随机删除一些现有边
    keep_mask = torch.rand(num_edges) > noise_ratio
    graph.edge_index = graph.edge_index[:, keep_mask]
    graph.edge_feature = graph.edge_feature[keep_mask]
    graph.edge_time = graph.edge_time[keep_mask]
    
    # 随机添加新边
    num_nodes = graph.node_feature.size(0)
    new_edges = torch.randint(0, num_nodes, (2, num_noise_edges))
    graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
    
    # 为新边生成随机特征和时间
    new_features = torch.rand(num_noise_edges, graph.edge_feature.size(1))
    graph.edge_feature = torch.cat([graph.edge_feature, new_features], dim=0)
    
    new_times = torch.ones(num_noise_edges) * graph.edge_time.mean()
    graph.edge_time = torch.cat([graph.edge_time, new_times])
    
    return graph


def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir)
    
    # 如果不需要快照，直接添加噪声并返回
    if not snapshot:
        if cfg.gnn.noise > 0.0:
            g_all = add_structural_noise(g_all, cfg.gnn.noise)
        return g_all

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = split_by_seconds(g_all, freq)
    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq)
    num_nodes = g_all.edge_index.max() + 1

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    # check snapshots ordering.
    prev_end = -1
    for g in snapshot_list:
        start, end = torch.min(g.edge_time), torch.max(g.edge_time)
        assert prev_end < start <= end
        prev_end = end

    # 在返回快照列表之前为每个快照添加噪声
    if cfg.gnn.noise > 0.0:
        snapshot_list = [add_structural_noise(g, cfg.gnn.noise) 
                        for g in snapshot_list]

    return snapshot_list

name_dict = {
    'bitcoin-alpha': 'bitcoinalpha.csv',
    'bitcoin-otc': 'bitcoinotc.csv',
    'er_5': 'er_5.csv',
    'er_20': 'er_20.csv',
    'er_50': 'er_50.csv',
    'er_90': 'er_90.csv',
}
    

def load_btc_dataset(path, name):
    graphs = load_generic(os.path.join(path, name_dict[name]),
                            snapshot=cfg.transaction.snapshot,
                            snapshot_freq=cfg.transaction.snapshot_freq)
    if cfg.dataset.split_method == 'chronological_temporal':
        return graphs
    else:
        # The default split (80-10-10) requires at least 10 edges each
        # snapshot.
        filtered_graphs = list()
        for g in graphs:
            if g.num_edges >= 10:
                filtered_graphs.append(g)
        return filtered_graphs

