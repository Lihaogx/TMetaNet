import os
import torch
from topo_utils.homology_utils import compute_persistence_image

def compute_topo_features(args):
    """计算拓扑特征
    
    Args:
        edge_sequences: 边序列列表
        args: 配置参数
        
    Returns:
        topo_features: 拓扑特征张量
    """
    topo_features_list = []

    window_size = args.topo.window_size
    remove_edge = args.topo.remove_edge
    is_directed = args.topo.is_directed
    path = "dataset/" + args.dataset + "/topo_feature/"
    
    for filtration in args.topo.filtration:
        epsilon, delta = filtration
        # 生成文件名
        filename = f"topo_diagram_w{window_size}_e{epsilon}_d{delta}_r{remove_edge}_d{is_directed}.pt"
        filepath = os.path.join(path, filename)
        
        # 检查文件是否存在
        if os.path.exists(filepath):
            print(f"从{filepath}加载缓存的拓扑图")
            topo_diagram = torch.load(filepath)
        else:
            print(f"无法计算{filename}的拓扑图")
            raise ValueError(f"未找到缓存的拓扑图文件{filepath}")
            
        # 计算0维和1维的persistence image
        diagram_features = [
            compute_persistence_image(dim, topo_diagram, [args.topo.resolution]*2, window_size, args.topo.bandwidth, args.topo.power)
            for dim in [0,1]
        ]
                
        # 转换为tensor
        topo_features = [torch.tensor(f, dtype=torch.float) for f in diagram_features]
        topo_features = torch.stack(topo_features, dim=1)
        topo_features_list.append(topo_features)
        
    # 将所有特征拼接在一起
    topo_features = torch.cat(topo_features_list, dim=1)
    return topo_features