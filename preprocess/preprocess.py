import os
import numpy as np
from pathlib import Path

class DataPreprocessor:
    def __init__(self, format_name):
        self.format_name = format_name
        self.data_path = "./preprocess/raw_data"
        self.output_path = "./dataset" 
        
        # Define dataset and loader mapping relationships
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
                'func': 'load_generic_dataset', 
                'format': 'reddit_hyperlink',
                'name': 'reddit-body.tsv'
            },
            'reddit-title': {
                'path': 'reddit-title',
                'loader': 'roland_reddit_hyperlink',
                'func': 'load_generic_dataset',
                'format': 'reddit_hyperlink',
                'name': 'reddit-title.tsv'
            },
        }
        
    def load_dataset(self):
        """Load corresponding dataset processing function based on format"""
        if self.format_name not in self.dataset_loader_map:
            raise ValueError(f"Unsupported dataset format: {self.format_name}")
            
        dataset_info = self.dataset_loader_map[self.format_name]
        loader_file = f"{dataset_info['loader']}.py"
        loader_path = os.path.join("./model/roland_data_loader", loader_file)
        
        if not os.path.exists(loader_path):
            raise ValueError(f"Loader file not found: {loader_path}")
            
        # Dynamically import corresponding loader module
        import importlib.util
        spec = importlib.util.spec_from_file_location(dataset_info['loader'], loader_path)
        loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader_module)
        
        # Get dataset path
        dataset_path = os.path.join(self.data_path, dataset_info['path'])
        
        # Call corresponding loader function
        load_func = getattr(loader_module, dataset_info['func'])
        graphs = load_func(dataset_path, format_name)
        
        return graphs
        
    def process_graphs(self, graphs):
        """Process graph data and save in specified format"""
        # Create output directory
        dataset_dir = os.path.join(self.output_path, self.format_name)
        subdirs = ['edge_feature', 'edge_index', 'edge_time', 'node_feature']
        
        for subdir in subdirs:
            Path(os.path.join(dataset_dir, subdir)).mkdir(parents=True, exist_ok=True)
            
        # Convert graphs to list if not already
        if not isinstance(graphs, list):
            graphs = [graphs]
            
        # Process each graph
        for i, graph in enumerate(graphs):
            # Save edge features
            if hasattr(graph, 'edge_feature'):
                edge_feature_path = os.path.join(dataset_dir, 'edge_feature', f'{i}.npy')
                np.save(edge_feature_path, graph.edge_feature)
                
            # Save edge indices
            edge_index_path = os.path.join(dataset_dir, 'edge_index', f'{i}.npy')
            np.save(edge_index_path, graph.edge_index)
            
            # Save edge timestamps (if they exist)
            if hasattr(graph, 'edge_time'):
                edge_time_path = os.path.join(dataset_dir, 'edge_time', f'{i}.npy')
                np.save(edge_time_path, graph.edge_time)
                
            # Save node features
            if hasattr(graph, 'node_feature'):
                node_feature_path = os.path.join(dataset_dir, 'node_feature', f'{i}.npy')
                np.save(node_feature_path, graph.node_feature)
                
    def run(self):
        """Run the entire preprocessing pipeline"""
        print(f"Starting to process {self.format_name} dataset...")
        graphs = self.load_dataset()
        self.process_graphs(graphs)
        print(f"Dataset processing completed, saved to {self.output_path}/{self.format_name}")

if __name__ == "__main__":
    # Process all datasets
    datasets = ['ethereum']
    
    for format_name in datasets:
        preprocessor = DataPreprocessor(format_name)
        preprocessor.run()
