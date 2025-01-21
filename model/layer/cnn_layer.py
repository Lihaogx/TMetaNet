import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg


class BaseCNN(nn.Module):
    """基础CNN类，其他CNN类可以继承这个类"""
    def __init__(self, dim_out):
        super(BaseCNN, self).__init__()
        self.dim_out = dim_out
        self.num_channels = 2 * len(cfg.topo.filtration)

    def _reshape_input(self, x):
        """处理输入形状"""
        original_shape = x.shape[:-3]
        x = x.view(-1, self.num_channels, 50, 50)
        return x, original_shape

    def _reshape_output(self, x, original_shape):
        """处理输出形状"""
        return x.view(*original_shape, self.dim_out)

class ImprovedFusionCNN(BaseCNN):
    """改进的FusionCNN，整合了之前的多个CNN实现"""
    def __init__(self, dim_out):
        super(ImprovedFusionCNN, self).__init__(dim_out)
        
        # 使用SE (Squeeze-and-Excitation) 模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.Sigmoid()
        )
        
        # 使用残差连接和SE注意力机制
        self.features = nn.Sequential(
            # 第一个块
            nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(2),
            
            # 第二个块 (带残差连接)
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            
            # 第三个块 (带残差连接)
            ResidualBlock(64, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 使用LayerNorm替代普通的全连接层
        self.fc = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, dim_out) if dim_out != 64 else nn.Identity(),
            nn.Dropout(p=cfg.topo.dropout)
        )
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """改进的权重初始化"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x, original_shape = self._reshape_input(x)
        
        # 特征提取
        features = self.features(x)
        
        # 应用SE注意力
        se_weight = self.se(features)
        features = features * se_weight
        
        # 展平并通过全连接层
        features = features.squeeze(-1).squeeze(-1)
        x = self.fc(features)
        
        return self._reshape_output(x, original_shape)

class ResidualBlock(nn.Module):
    """改进的残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
        self.dropout = nn.Dropout2d(p=cfg.topo.dropout)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += residual
        out = F.relu(out)
        return out
    
class DilatedResidualFusionCNN(nn.Module):
    def __init__(self, dim_out):
        super(DilatedResidualFusionCNN, self).__init__()
        self.num_channels = 2 * len(cfg.topo.filtration)
        self.dim_out = dim_out

        self.features = nn.Sequential(
            DilatedResidualBlock(self.num_channels, 32),
            nn.MaxPool2d(2),

            DilatedResidualBlock(32, 64),
            nn.MaxPool2d(2),

            DilatedResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=cfg.topo.dropout),
            nn.Linear(64, dim_out) if dim_out != 64 else nn.Identity()
        )

    def forward(self, x):
        original_shape = x.shape[:-3]
        x = x.view(-1, self.num_channels, 50, 50)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(*original_shape, self.dim_out)
        return x

# Dilated Residual Block Definition
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out
class ResidualFusionCNN(nn.Module):
    def __init__(self, dim_out):
        super(ResidualFusionCNN, self).__init__()
        self.num_channels = 2 * len(cfg.topo.filtration)
        self.dim_out = dim_out

        # 主干网络
        self.features = nn.Sequential(
            DilatedResidualBlock(self.num_channels, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=cfg.topo.dropout),

            DilatedResidualBlock(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=cfg.topo.dropout),

            DilatedResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(p=cfg.topo.dropout)
        )

        # 修改多尺度特征提取分支，使输出通道数匹配
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                DilatedResidualBlock(self.num_channels, 32),
                DilatedResidualBlock(32, 128)
            )
            for _ in range(3)
        ])
        
        # SE注意力模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // 16, 1),
            nn.ReLU(),
            nn.Conv2d(128 // 16, 128, 1),
            nn.Sigmoid()
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=cfg.topo.dropout),
            nn.Linear(64, dim_out) if dim_out != 64 else nn.Identity()
        )

    def forward(self, x):
        original_shape = x.shape[:-3]
        x = x.view(-1, self.num_channels, 50, 50)
        
        # 主干特征
        main_features = self.features(x)  # [B, 128, 1, 1]
        
        # 多尺度特征融合
        multi_scale_features = []
        for conv in self.multi_scale:
            scale_feat = conv(x)  # 现在每个分支输出128通道
            scale_feat = F.adaptive_avg_pool2d(scale_feat, (1, 1))
            multi_scale_features.append(scale_feat)
        
        # 特征融合 - 现在所有特征都是128通道
        multi_scale_features = torch.mean(torch.stack(multi_scale_features), dim=0)  # 使用平均值而不是拼接
        
        # SE注意力
        se_weight = self.se(main_features)
        main_features = main_features * se_weight
        
        # 特征整合 - 添加残差连接
        features = main_features + multi_scale_features  # 主干特征和多尺度特征的残差连接
        features = F.relu(features)  # 激活函数
        features = features.view(features.size(0), -1)
        
        # 全连接层
        out = self.fc(features)
        out = out.view(*original_shape, self.dim_out)
        
        return out



class AttentionFusionCNN(nn.Module):
    def __init__(self, dim_out):
        super(AttentionFusionCNN, self).__init__()
        self.num_channels = 2 * len(cfg.topo.filtration)
        self.dim_out = dim_out

        self.features = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(64, dim_out) if dim_out != 64 else nn.Identity()

    def forward(self, x):
        original_shape = x.shape[:-3]
        x = x.view(-1, self.num_channels, 50, 50)
        features = self.features(x)  # [batch, 64, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 64]

        # Attention mechanism
        attention_weights = self.attention(features)  # [batch, 1]
        features = features * attention_weights  # [batch, 64]

        x = self.fc(features)  # [batch, dim_out]
        x = x.view(*original_shape, self.dim_out)
        return x


class CNN(nn.Module):
    def __init__(self, dim_out):
        super(CNN, self).__init__()
        self.dim_out = dim_out
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, dim_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.resolution = cfg.topo.resolution

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，最后两维应该是50x50
                支持形状: [..., 50, 50]
        Returns:
            torch.Tensor: 形状为[..., dim_out]的特征张量
        """
        # 保存原始batch维度
        original_shape = x.shape[:-2]
        # 将所有batch维度合并
        x = x.view(-1, 1, self.resolution, self.resolution)
        # 通过CNN
        feature = self.features(x)  # -> [-1, dim_out, 1, 1]
        # 移除最后两个1维度
        feature = feature.squeeze(-1).squeeze(-1)  # -> [-1, dim_out]
        # 恢复原始batch维度
        feature = feature.view(*original_shape, self.dim_out)
        return feature
    
class MultiChannelCNN(nn.Module):
    def __init__(self, dim_out):
        super(MultiChannelCNN, self).__init__()
        self.num_channels = 2 * len(cfg.topo.filtration)  # 输入通道数为filtration长度的2倍
        
        self.features = nn.Sequential(
            # 输入是num_channels个50x50的图,每个图作为一个通道
            nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1),  # [batch, num_channels, 50, 50] -> [batch, 32, 50, 50]
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(2),  # -> [batch, 32, 25, 25]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [batch, 64, 25, 25]
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(2),  # -> [batch, 64, 12, 12]
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # -> [batch, 64, 12, 12]
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [batch, 64, 1, 1]
        )
        
        self.fc = nn.Linear(64, dim_out) if dim_out != 64 else nn.Identity()
        
    def forward(self, x):
        """处理多通道拓扑特征图
        Args:
            x (torch.Tensor): 多通道拓扑图,形状为[num_channels, 50, 50]
        Returns:
            torch.Tensor: 融合后的特征 [dim_out]
        """
        # 检查输入通道数是否正确
        assert x.shape[0] == self.num_channels, \
            f"Expected {self.num_channels} input channels, but got {x.shape[0]}"
            
        x = x.unsqueeze(0)  # [1, num_channels, 50, 50]
        
        x = self.features(x)  # [1, 64, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [1, 64]
        x = self.fc(x)  # [1, dim_out]
        return x.squeeze(0)  # [dim_out]
    


class FusionCNN(nn.Module):
    def __init__(self, dim_out):
        super(FusionCNN, self).__init__()
        self.num_channels = 2 * len(cfg.topo.filtration)  # 输入通道数为filtration长度的2倍
        self.dim_out = dim_out
        
        self.features = nn.Sequential(
            # 输入是num_channels个50x50的图,每个图作为一个通道
            nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1),  # [batch, num_channels, 50, 50] -> [batch, 32, 50, 50]
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(2),  # -> [batch, 32, 25, 25]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [batch, 64, 25, 25]
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.MaxPool2d(2),  # -> [batch, 64, 12, 12]
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # -> [batch, 64, 12, 12]
            nn.ReLU(),
            nn.Dropout2d(p=cfg.topo.dropout),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [batch, 64, 1, 1]
        )
        
        self.fc = nn.Linear(64, dim_out) if dim_out != 64 else nn.Identity()
        
    def forward(self, x):
        """处理多通道拓扑特征图
        Args:
            x (torch.Tensor): 多通道拓扑图，形状为[..., num_channels, 50, 50]
        Returns:
            torch.Tensor: 融合后的特征 [..., dim_out]
        """
        # 保存原始batch维度
        original_shape = x.shape[:-3]
        
        # 将所有batch维度合并
        x = x.view(-1, self.num_channels, 50, 50)
        
        x = self.features(x)  # [-1, 64, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [-1, 64]
        x = self.fc(x)  # [-1, dim_out]
        
        # 恢复原始batch维度
        x = x.view(*original_shape, self.dim_out)
        return x