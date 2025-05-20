import os
import torch
from topo_utils.homology_utils import compute_persistence_image

def compute_topo_features(args):
    """Compute topological features
    
    Args:
        edge_sequences: List of edge sequences
        args: Configuration parameters
        
    Returns:
        topo_features: Topological feature tensor
    """
    topo_features_list = []

    window_size = args.topo.window_size
    remove_edge = args.topo.remove_edge
    is_directed = args.topo.is_directed
    path = "dataset/" + args.dataset + "/topo_feature/"
    
    for filtration in args.topo.filtration:
        epsilon, delta = filtration
        # Generate filename
        filename = f"topo_diagram_w{window_size}_e{epsilon}_d{delta}_r{remove_edge}_d{is_directed}.pt"
        filepath = os.path.join(path, filename)
        
        # Check if file exists
        if os.path.exists(filepath):
            print(f"Loading cached topology diagram from {filepath}")
            topo_diagram = torch.load(filepath)
        else:
            print(f"Cannot compute topology diagram for {filename}")
            raise ValueError(f"Cached topology diagram file not found: {filepath}")
            
        # Compute persistence image for 0D and 1D
        diagram_features = [
            compute_persistence_image(dim, topo_diagram, [args.topo.resolution]*2, window_size, args.topo.bandwidth, args.topo.power)
            for dim in [0,1]
        ]
                
        # Convert to tensor
        topo_features = [torch.tensor(f, dtype=torch.float) for f in diagram_features]
        topo_features = torch.stack(topo_features, dim=1)
        topo_features_list.append(topo_features)
        
    # Concatenate all features
    topo_features = torch.cat(topo_features_list, dim=1)
    return topo_features