# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusionModule(nn.Module):
    """
    自适应融合模块 (Adaptive Fusion Module, AF)。
    该模块借鉴自 GLF-Net，用于动态融合来自两个不同编码器的特征图。
    """
    def __init__(self, in_channels):
        """
        初始化自适应融合模块。
        Args:
            in_channels (int): 输入特征图的通道数。
        """
        super(AdaptiveFusionModule, self).__init__()

        # 1x1 卷积，用于在拼接后降低通道数，并提取初步的融合信息
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

        # 3x3 卷积，用于在融合信息的基础上生成空间注意力门控
        self.gate_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x_sam, x_mamba):
        """
        前向传播函数。
        Args:
            x_sam (torch.Tensor): 来自 SAM/Hiera 分支的特征图。
            x_mamba (torch.Tensor): 来自 SAVSS/Mamba 分支的特征图。
        Returns:
            torch.Tensor: 融合后的特征图，维度与输入相同。
        """
        # 确保两个输入的维度一致
        assert x_sam.shape == x_mamba.shape, "输入到 AdaptiveFusionModule 的两个特征图必须具有相同的维度"

        # 1. 拼接 (Concat)
        # 在通道维度上拼接两个特征图
        x_cat = torch.cat([x_sam, x_mamba], dim=1) # 形状: [B, C*2, H, W]

        # 2. 初步融合 (1x1 Conv)
        # 通过 1x1 卷积将通道数恢复到 in_channels
        x_fused = self.fusion_conv(x_cat) # 形状: [B, C, H, W]

        # 3. 生成门控信号 (3x3 Conv -> Sigmoid)
        # 使用 3x3 卷积提取上下文信息，并用 Sigmoid 生成一个范围在 (0, 1) 之间的门控权重
        gate = torch.sigmoid(self.gate_conv(x_fused)) # 形状: [B, C, H, W]

        # 4. 门控融合 (Gated Fusion)
        # 使用门控权重来动态地、逐像素地融合两个原始输入特征
        # gate 权重较高的地方，更多地保留 x_sam 的信息
        # (1 - gate) 权重较高的地方，更多地保留 x_mamba 的信息
        output = x_sam * gate + x_mamba * (1 - gate)

        return output

# --- 任务1: 创建 LightAdaptiveFusionModule ---

class LightAdaptiveFusionModule(nn.Module):
    """
    轻量化自适应融合模块 (Lightweight Adaptive Fusion Module)。
    该模块通过引入瓶颈设计来显著减少生成门控信号路径上的参数量。
    作者: Roy
    """
    def __init__(self, in_channels, reduction_ratio=4):
        """
        初始化轻量化自适应融合模块。
        Args:
            in_channels (int): 输入特征图的通道数。
            reduction_ratio (int): 瓶颈设计的通道缩减率。
        """
        super(LightAdaptiveFusionModule, self).__init__()
        
        # 计算瓶颈部分的通道数
        bottleneck_channels = in_channels // reduction_ratio

        # 用于生成门控信号 alpha 的路径，采用瓶颈设计
        self.attention_generator = nn.Sequential(
            # 1. 降维: 1x1 卷积将通道数从 in_channels*2 压缩到 bottleneck_channels
            nn.Conv2d(in_channels * 2, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            
            # 2. 特征处理: 在低维空间中使用 3x3 卷积进行特征提取
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),

            # 3. 升维: 1x1 卷积将通道数从 bottleneck_channels 恢复到 in_channels
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, bias=False),
            
            # 4. Sigmoid: 生成最终的门控权重
            nn.Sigmoid()
        )

    def forward(self, x_sam, x_mamba):
        """
        前向传播函数。
        Args:
            x_sam (torch.Tensor): 来自 SAM/Hiera 分支的特征图。
            x_mamba (torch.Tensor): 来自 SAVSS/Mamba 分支的特征图。
        Returns:
            torch.Tensor: 融合后的特征图，维度与输入相同。
        """
        # 确保两个输入的维度一致
        assert x_sam.shape == x_mamba.shape, "输入到 LightAdaptiveFusionModule 的两个特征图必须具有相同的维度"

        # 1. 拼接 (Concat)
        # 在通道维度上拼接两个特征图
        x_cat = torch.cat([x_sam, x_mamba], dim=1) # 形状: [B, C*2, H, W]

        # 2. 生成门控信号 (Attention Generator)
        # 通过带有瓶颈设计的路径生成门控权重 alpha
        gate = self.attention_generator(x_cat) # 形状: [B, C, H, W]

        # 3. 门控融合 (Gated Fusion)
        # 使用门控权重来动态地、逐像素地融合两个原始输入特征
        output = x_sam * gate + x_mamba * (1 - gate)

        return output
