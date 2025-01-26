import os
import numpy as np
from pathlib import Path
import torch

class DataPreprocessor:
    def __init__(self, format_name):
        self.format_name = format_name
        self.data_path = "/home/lh/Dowzag_2.0/preprocess/data"
        self.output_path = "/home/lh/Dowzag_2.0/dataset" 
        
        # 定义数据集和loader的映射关系
        self.dataset_loader_map = {
            'ethereum': {
                'path': 'ethereum',
                'loader': 'roland_eth',
                'func': 'load_eth_dataset',
                'format': 'ethereum',
                'name': 'networkyocoinTX.txt'
            },
            'as': {
                'path': 'as-733',
                'loader': 'roland_as',
                'func': 'load_generic_dataset',
                'format': 'as',
                'name': 'as.txt'
            },
            'bitcoin-alpha': {
                'path': 'bitcoin-alpha', 
                'loader': 'roland_btc',
                'func': 'load_btc_dataset',
                'format': 'bitcoin',
                'name': 'bitcoinalpha.csv'
            },
            'bitcoin-otc': {
                'path': 'bitcoin-otc',
                'loader': 'roland_btc', 
                'func': 'load_btc_dataset',
                'format': 'bitcoin',
                'name': 'bitcoinotc.csv'
            },
            'reddit-body': {
                'path': 'reddit-body',
                'loader': 'roland_reddit_hyperlink',
                'func': 'load_reddit_dataset', 
                'format': 'reddit_hyperlink',
                'name': 'reddit-body.tsv'
            },
            'reddit-title': {
                'path': 'reddit-title',
                'loader': 'roland_reddit_hyperlink',
                'func': 'load_reddit_dataset',
                'format': 'reddit_hyperlink',
                'name': 'reddit-title.tsv'
            },
            'uci-msg': {
                'path': 'uci-msg',
                'loader': 'roland_ucimsg',
                'func': 'load_uci_dataset',
                'format': 'ucimsg',
                'name': 'CollegeMsg.txt'
            },
        }
        
    def load_dataset(self):
        """根据format加载对应的数据集处理函数"""
        if self.format_name not in self.dataset_loader_map:
            raise ValueError(f"不支持的数据集格式: {self.format_name}")
            
        dataset_info = self.dataset_loader_map[self.format_name]
        loader_file = f"{dataset_info['loader']}.py"
        loader_path = os.path.join("/home/lh/Dowzag_2.0/model/roland_data_loader", loader_file)
        
        if not os.path.exists(loader_path):
            raise ValueError(f"找不到loader文件: {loader_path}")
            
        # 动态导入对应的loader模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(dataset_info['loader'], loader_path)
        loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader_module)
        
        # 获取数据集路径
        dataset_path = os.path.join(self.data_path, dataset_info['path'])
        
        # 调用对应的loader函数
        load_func = getattr(loader_module, dataset_info['func'])
        graphs = load_func(dataset_path, format_name)
        
        return graphs
        
    def process_graphs(self, graphs, output_path=None):
        """处理图数据并保存为规定格式"""
        # 创建输出目录
        if output_path is None:
            output_path = os.path.join(self.output_path, self.format_name)
        subdirs = ['edge_feature', 'edge_index', 'edge_time', 'node_feature']
        
        for subdir in subdirs:
            Path(os.path.join(output_path, subdir)).mkdir(parents=True, exist_ok=True)
            
        # 如果graphs不是列表,转换为列表
        if not isinstance(graphs, list):
            graphs = [graphs]
            
        # 处理每个图
        for i, graph in enumerate(graphs):
            # 保存边特征
            if hasattr(graph, 'edge_feature'):
                edge_feature_path = os.path.join(output_path, 'edge_feature', f'{i}.npy')
                np.save(edge_feature_path, graph.edge_feature)
                
            # 保存边索引
            edge_index_path = os.path.join(output_path, 'edge_index', f'{i}.npy')
            np.save(edge_index_path, graph.edge_index)
            
            # 保存边时间戳（如果存在）
            if hasattr(graph, 'edge_time'):
                edge_time_path = os.path.join(output_path, 'edge_time', f'{i}.npy')
                np.save(edge_time_path, graph.edge_time)
                
            # 保存节点特征
            if hasattr(graph, 'node_feature'):
                node_feature_path = os.path.join(output_path, 'node_feature', f'{i}.npy')
                np.save(node_feature_path, graph.node_feature)
                
    def add_noise(self, graphs, mode, noise_ratio):
        """添加噪声到图中
        Args:
            graphs: 图列表
            mode: 噪声模式 ('poison' 或 'escape')
            noise_ratio: 噪声概率 (0.1, 0.2, 或 0.5)
        """
        start_idx = 0
        if mode == 'escape':
            # 逃逸模式只对后30%的图添加噪声
            start_idx = int(len(graphs) * 0.7)
        
        for i in range(start_idx, len(graphs)):
            graph = graphs[i]
            
            # 获取现有的边
            edge_index = graph.edge_index
            num_nodes = max(edge_index.max(), edge_index.min()) + 1
            
            # 计算需要翻转的边数量
            num_flip_edges = int(edge_index.size(1) * noise_ratio)
            
            # 构建现有边的集合
            existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
            
            # 随机选择要删除的现有边
            num_remove = num_flip_edges // 2
            edges_to_remove = torch.randperm(edge_index.size(1))[:num_remove]
            new_edge_index = torch.cat([edge_index[:, i].unsqueeze(1) for i in range(edge_index.size(1)) if i not in edges_to_remove], dim=1)
            
            # 添加新边
            num_add = num_flip_edges - num_remove
            new_edges = []
            while len(new_edges) < num_add:
                src = torch.randint(0, num_nodes, (1,)).item()
                dst = torch.randint(0, num_nodes, (1,)).item()
                edge = (src, dst)
                # 确保添加的是原图中不存在的边
                if src != dst and edge not in existing_edges:
                    new_edges.append(edge)
                    
            # 将新边添加到图中
            new_edges = torch.tensor(new_edges, dtype=torch.long).t()
            graph.edge_index = torch.cat([new_edge_index, new_edges], dim=1)
            
            graphs[i] = graph
        
        return graphs

    def run(self):
        """运行整个预处理流程"""
        print(f"开始处理 {self.format_name} 数据集...")
        graphs = self.load_dataset()
        
        # 添加不同模式和比例的噪声
        modes = ['poison', 'escape']
        ratios = [0.05, 0.1, 0.2, 0.3]
        
        for mode in modes:
            for ratio in ratios:
                noisy_graphs = self.add_noise(graphs.copy(), mode, ratio)
                # 更新输出路径和格式名称
                noise_format = f"{self.format_name}_{mode}_{int(ratio*100)}"
                noise_output = os.path.join(self.output_path, noise_format)
                
                # 确保输出目录存在
                os.makedirs(noise_output, exist_ok=True)
                
                # 处理并保存带噪声的图
                self.process_graphs(noisy_graphs, output_path=noise_output)
                print(f"已保存带{ratio*100}%噪声的{mode}模式数据到 {noise_output}")
        
        # 处理原始数据集
        self.process_graphs(graphs)
        print(f"原始数据集处理完成，已保存到 {self.output_path}/{self.format_name}")

if __name__ == "__main__":
    # 遍历所有数据集进行处理
    datasets = ['reddit-title']
    
    for format_name in datasets:
        preprocessor = DataPreprocessor(format_name)
        preprocessor.run()
