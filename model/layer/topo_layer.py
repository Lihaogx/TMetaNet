import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, gnn_dim, cnn_dim, combined_dim, cfg):
        super(GatedFusion, self).__init__()
        self.gate = nn.Linear(gnn_dim + cnn_dim, 1)
        self.fc = nn.Linear(gnn_dim + (gnn_dim + cnn_dim), combined_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.topo.dropout)
    
    def forward(self, gnn_embedding, cnn_embedding):
        num_nodes = gnn_embedding.shape[0]
        if len(cnn_embedding.shape) == 1:
            cnn_embedding = cnn_embedding.unsqueeze(0)
        elif len(cnn_embedding.shape) == 2 and cnn_embedding.shape[0] != num_nodes:
            cnn_embedding = cnn_embedding[0].unsqueeze(0)
        cnn_embedding = cnn_embedding.expand(num_nodes, -1)
        combined_input = torch.cat([gnn_embedding, cnn_embedding], dim=1)
        gate = torch.sigmoid(self.gate(combined_input))
        fused = gate * gnn_embedding + (1 - gate) * cnn_embedding
        final_input = torch.cat([fused, combined_input], dim=1)
        output = self.fc(final_input)
        output = self.activation(output)
        output = self.dropout(output)
        return output
    
    

class AttentionFusion(nn.Module):
    def __init__(self, gnn_dim, cnn_dim, combined_dim, cfg):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(gnn_dim, combined_dim)
        self.key = nn.Linear(cnn_dim, combined_dim)
        self.value = nn.Linear(cnn_dim, combined_dim)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.topo.dropout)
    
    def forward(self, gnn_embedding, cnn_embedding):
        # gnn_embedding: [gnn_dim]
        # cnn_embedding: [cnn_dim]
        gnn_proj = self.query(gnn_embedding)  # [1, combined_dim]
        cnn_proj = self.key(cnn_embedding).unsqueeze(0)    # [1, combined_dim]
        value = self.value(cnn_embedding).unsqueeze(0)     # [1, combined_dim]
        
        attention_scores = torch.matmul(gnn_proj, cnn_proj.transpose(0, 1))  # [1, 1]
        attention_weights = self.softmax(attention_scores)  # [1, 1]
        fused = attention_weights * value + gnn_proj.squeeze(0)  # [combined_dim]
        fused = self.activation(fused)
        fused = self.dropout(fused)
        return fused

class ElementWiseMultiplicationFusion(nn.Module):
    def __init__(self, gnn_dim, cnn_dim, combined_dim, cfg):
        super(ElementWiseMultiplicationFusion, self).__init__()
        # 修改: 如果 gnn_dim 和 cnn_dim 不同，需要映射到相同的维度
        if gnn_dim != cnn_dim:
            self.gnn_proj = nn.Linear(gnn_dim, combined_dim)
            self.cnn_proj = nn.Linear(cnn_dim, combined_dim)
        else:
            self.gnn_proj = None
            self.cnn_proj = None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.topo.dropout)
    
    def forward(self, gnn_embedding, cnn_embedding):
        num_nodes = gnn_embedding.shape[0]
        # 将CNN特征广播到所有节点
        cnn_embedding = cnn_embedding.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, cnn_dim]
        
        if self.gnn_proj and self.cnn_proj:
            gnn_proj = self.gnn_proj(gnn_embedding)  # [num_nodes, combined_dim]
            cnn_proj = self.cnn_proj(cnn_embedding)  # [num_nodes, combined_dim]
        else:
            gnn_proj = gnn_embedding
            cnn_proj = cnn_embedding
            
        combined = gnn_proj * cnn_proj  # [num_nodes, combined_dim]
        combined = self.activation(combined)
        combined = self.dropout(combined)
        return combined
    
class ElementWiseAdditionFusion(nn.Module):
    def __init__(self, gnn_dim, cnn_dim, combined_dim, cfg):
        super(ElementWiseAdditionFusion, self).__init__()
        if gnn_dim != cnn_dim:
            self.gnn_proj = nn.Linear(gnn_dim, combined_dim)
            self.cnn_proj = nn.Linear(cnn_dim, combined_dim)
        else:
            self.gnn_proj = None
            self.cnn_proj = None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.topo.dropout)
    
    def forward(self, gnn_embedding, cnn_embedding):
        if self.gnn_proj and self.cnn_proj:
            gnn_proj = self.gnn_proj(gnn_embedding)  # [combined_dim]
            cnn_proj = self.cnn_proj(cnn_embedding)  # [combined_dim]
        else:
            gnn_proj = gnn_embedding
            cnn_proj = cnn_embedding
        combined = gnn_proj + cnn_proj  # [combined_dim]
        combined = self.activation(combined)
        combined = self.dropout(combined)
        return combined
    
    
class ConcatenationFusion(nn.Module):
    def __init__(self, gnn_dim, cnn_dim, combined_dim, cfg):
        super(ConcatenationFusion, self).__init__()
        self.fc = nn.Linear(gnn_dim + cnn_dim, combined_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.topo.dropout)
    
    def forward(self, gnn_embedding, cnn_embedding):
        num_nodes = gnn_embedding.shape[0]  # 11762
        
        # 修改: 调整CNN特征的维度以匹配GNN特征的节点数
        if len(cnn_embedding.shape) == 1:
            # 如果CNN特征是1维的,先扩展为2维
            cnn_embedding = cnn_embedding.unsqueeze(0)  # [1, cnn_dim]
        
        # 将CNN特征广播到所有节点
        cnn_embedding = cnn_embedding.expand(num_nodes, -1)  # [num_nodes, cnn_dim]
        
        # 在特征维度上拼接
        combined = torch.cat([gnn_embedding, cnn_embedding], dim=1)  # [num_nodes, gnn_dim + cnn_dim]
        combined = self.fc(combined)  # [num_nodes, combined_dim]
        combined = self.activation(combined)
        combined = self.dropout(combined)
        return combined

class BilinearFusion(nn.Module):
    def __init__(self, gnn_dim, cnn_dim, combined_dim, cfg):
        super(BilinearFusion, self).__init__()
        self.bilinear = nn.Bilinear(gnn_dim, cnn_dim, combined_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.topo.dropout)
    
    def forward(self, gnn_embedding, cnn_embedding):
        num_nodes = gnn_embedding.shape[0]
        cnn_embedding = cnn_embedding.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, cnn_dim]
        

        fused = self.bilinear(gnn_embedding, cnn_embedding)  # [num_nodes, combined_dim]
        fused = self.activation(fused)
        fused = self.dropout(fused)
        return fused

class FiLMFusion(nn.Module):
    def __init__(self, gnn_dim, cnn_dim, combined_dim, cfg):
        super(FiLMFusion, self).__init__()
        self.gamma = nn.Linear(gnn_dim, combined_dim)
        self.beta = nn.Linear(gnn_dim, combined_dim)
        self.cnn_proj = nn.Linear(cnn_dim, combined_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.topo.dropout)
    
    def forward(self, gnn_embedding, cnn_embedding):
        num_nodes = gnn_embedding.shape[0]
        # 将CNN特征广播到所有节点
        cnn_embedding = cnn_embedding.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, cnn_dim]
        
        gamma = self.gamma(gnn_embedding)  # [num_nodes, combined_dim]
        beta = self.beta(gnn_embedding)    # [num_nodes, combined_dim]
        cnn_proj = self.cnn_proj(cnn_embedding)  # [num_nodes, combined_dim]
        transformed = gamma * cnn_proj + beta  # [num_nodes, combined_dim]
        combined = self.activation(transformed)
        combined = self.dropout(combined)
        return combined
