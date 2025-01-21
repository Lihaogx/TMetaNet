import os
from typing import List, Union

import deepsnap
import numpy as np
import types
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
        
cfg_dict = {
    "transaction": {
        "snapshot": True,
        "snapshot_freq": '1200000s',
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

def load_single_dataset(dataset_path: str) -> Graph:
    df_trans = pd.read_csv(dataset_path, sep=' ', header=None, index_col=None)
    df_trans.columns = ['SOURCE', 'TARGET', 'TIME', 'VALUE']  

    num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

    df_trans['TIME'] = df_trans['TIME'].astype(np.int64).astype(np.float64)
    assert not np.any(pd.isna(df_trans).values)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIME'].values.reshape(-1, 1))
    
    value_scaler = MinMaxScaler(feature_range=(0, 1))
    df_trans['VALUE_SCALED'] = value_scaler.fit_transform(
        df_trans['VALUE'].values.reshape(-1, 1))
    edge_feature = torch.Tensor(
        df_trans[['VALUE_SCALED','TimestampScaled']].values)  # 直接按 Wei 单位保留

    node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
    enc = OrdinalEncoder(categories=[node_indices, node_indices])
    raw_edges = df_trans[['SOURCE', 'TARGET']].values
    edge_index = enc.fit_transform(raw_edges).transpose()
    edge_index = torch.LongTensor(edge_index)


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

    freq_map = {'D': '%j',  # day of year
                'W': '%W',  # week of year
                'M': '%m'}  # month of year

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()

    for p in periods:
        # 获取时间段内的边索引
        period_members = period2id[p]
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


def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir)
    if not snapshot:
        return g_all

    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = split_by_seconds(g_all, freq)
    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq)

    num_nodes = g_all.edge_index.max() + 1

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    prev_end = -1
    for g in snapshot_list:
        start, end = torch.min(g.edge_time), torch.max(g.edge_time)
        assert prev_end < start <= end
        prev_end = end

    return snapshot_list

name_dict = {
    'ethereum': 'networktronixTX.txt',
}

def load_eth_dataset(path, name):
    graphs = load_generic(os.path.join(path, name_dict[name]),
                            snapshot=cfg.transaction.snapshot,
                            snapshot_freq=cfg.transaction.snapshot_freq)
    # if cfg.dataset.split_method == 'chronological_temporal':
    #     return graphs
    # else:
        # The default split (80-10-10) requires at least 10 edges each
        # snapshot.
    filtered_graphs = list()
    for g in graphs:
        if g.num_edges >= 10:
            filtered_graphs.append(g)
    return filtered_graphs
