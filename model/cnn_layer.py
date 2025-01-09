import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, dim_out, cfg):
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
        x = x.view(-1, 1, 50, 50)
        # 通过CNN
        feature = self.features(x)  # -> [-1, dim_out, 1, 1]
        # 移除最后两个1维度
        feature = feature.squeeze(-1).squeeze(-1)  # -> [-1, dim_out]
        # 恢复原始batch维度
        feature = feature.view(*original_shape, self.dim_out)
        return feature
    
class MultiChannelCNN(nn.Module):
    def __init__(self, dim_out, cfg):
        super(MultiChannelCNN, self).__init__()
        self.num_channels = 2 * len(cfg.topo.filtration)  # 输入通道数为filtration长度的2倍

        # 定义每个卷积层
        self.conv1 = nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 激活函数和池化层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=cfg.topo.dropout)
        self.pool = nn.MaxPool2d(2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层，如果dim_out != 64
        self.fc = nn.Linear(64, dim_out) if dim_out != 64 else nn.Identity()

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.conv1.weight, gain=gain)
        nn.init.xavier_normal_(self.conv2.weight, gain=gain)
        nn.init.xavier_normal_(self.conv3.weight, gain=gain)

        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)

        if not isinstance(self.fc, nn.Identity):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, fast_weights=None):
        """
        处理多通道拓扑特征图

        Args:
            x (torch.Tensor): 多通道拓扑图,形状为[num_channels, 50, 50]
            fast_weights (list of torch.Tensor, optional): 
                [conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, conv3_bias]

        Returns:
            torch.Tensor: 融合后的特征 [dim_out]
        """
        # 检查输入通道数是否正确
        assert x.shape[0] == self.num_channels, \
            f"Expected {self.num_channels} input channels, but got {x.shape[0]}"

        x = x.unsqueeze(0)  # [1, num_channels, 50, 50]

        if fast_weights:
            # 确保 fast_weights 的长度正确
            assert len(fast_weights) == 6, f"Expected 6 fast_weights, but got {len(fast_weights)}"

            # 解包 fast_weights
            conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, conv3_bias = fast_weights

            # 卷积层1
            x = F.conv2d(x, conv1_weight, conv1_bias, stride=1, padding=1)  # [1,32,50,50]
            x = self.relu(x)
            x = F.dropout2d(x, p=self.dropout.p, training=self.training)
            x = self.pool(x)  # [1,32,25,25]

            # 卷积层2
            x = F.conv2d(x, conv2_weight, conv2_bias, stride=1, padding=1)  # [1,64,25,25]
            x = self.relu(x)
            x = F.dropout2d(x, p=self.dropout.p, training=self.training)
            x = self.pool(x)  # [1,64,12,12]

            # 卷积层3
            x = F.conv2d(x, conv3_weight, conv3_bias, stride=1, padding=1)  # [1,64,12,12]
            x = self.relu(x)
            x = F.dropout2d(x, p=self.dropout.p, training=self.training)
            x = self.avg_pool(x)  # [1,64,1,1]
        else:
            # 使用定义好的卷积层
            x = self.conv1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.pool(x)

            x = self.conv3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.avg_pool(x)

        x = x.squeeze(-1).squeeze(-1)  # [1,64]

        x = self.fc(x)  # [1, dim_out] 或 [1,64]

        return x.squeeze(0)  # [dim_out]
    


class FusionCNN(nn.Module):
    def __init__(self, dim_out, cfg):
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