# Author: Roy
# Copyright (c) Your Company. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is revised based on the detailed description in the SAMba-UNet paper.
# According to user requirements, this file has been refactored for lightweighting.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAdapter(nn.Module):
    """A simple MLP adapter module.

    This module is used to enhance the non-linear capabilities of the features.
    The architecture is a simple two-layer MLP with a GeLU activation in between.
    It is used in parallel with the DynamicFeatureFusionRefiner.
    """

    def __init__(self, dim, mlp_ratio=2.0, drop=0.1):
        """
        Args:
            dim (int): Input and output dimension.
            mlp_ratio (float): The ratio to determine the hidden dimension size.
            drop (float): Dropout rate.
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass for the MLP adapter.
        Input shape: (B, C, H, W)
        The MLP operates on the channel dimension, so we permute.
        """
        # Permute to (B, H, W, C) for Linear layer
        x_permuted = x.permute(0, 2, 3, 1)
        
        out = self.fc1(x_permuted)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        
        # Permute back to (B, C, H, W)
        out = out.permute(0, 3, 1, 2)
        return out


class DynamicFeatureFusionRefiner(nn.Module):
    """Dynamic Feature Fusion Refiner (DFFR).

    This module is implemented strictly following the logic described in the
    SAMba-UNet paper (Section 3.2, Figure 2).
    It refines features to bridge the domain gap for medical images.
    
    **Lightweight Refactoring**: This module has been refactored to significantly
    reduce parameter count by introducing bottleneck designs in both the channel
    attention and spatial enhancement paths.
    """

    # 根据用户要求，重构 __init__ 方法以实现轻量化
    def __init__(self, dim, reduction_ratio=16, spatial_channels_ratio=4, spatial_kernel_size=3):
        """
        Args:
            dim (int): The number of channels in the input feature map (C).
            reduction_ratio (int): The bottleneck compression ratio for channel attention.
                                   A higher value means more parameter reduction.
            spatial_channels_ratio (int): The bottleneck ratio for the spatial enhancement path.
            spatial_kernel_size (int): The kernel size for the spatial enhancement convs.
        """
        super().__init__()
        self.dim = dim
        padding = spatial_kernel_size // 2

        # --- 通道注意力瓶颈设计 (Channel Attention Bottleneck) ---
        # 根据用户要求，使用 reduction_ratio 来计算瓶颈维度，大幅减少 MLP 的参数。
        # 例如，如果 dim=256, reduction_ratio=16, 则 bottleneck_dim=16。
        bottleneck_dim = int(dim / reduction_ratio)

        # MLP 的输入维度是 dim * 2，因为拼接了平均池化和最大池化的结果。
        # 结构: Linear(dim*2 -> bottleneck_dim) -> ReLU -> Linear(bottleneck_dim -> dim)
        self.channel_mlp_w1 = nn.Linear(dim * 2, bottleneck_dim)
        self.channel_mlp_w2 = nn.Linear(bottleneck_dim, dim)

        # --- 空间增强路径瓶颈设计 (Spatial Enhancement Bottleneck) ---
        # 根据用户要求，在空间路径中同样引入瓶颈，减少卷积层的参数量。
        spatial_bottleneck_dim = int(dim / spatial_channels_ratio)
        
        # 原始结构: Conv2d(dim -> dim) -> ConvTranspose2d(dim -> dim)
        # 修改后结构: Conv2d(dim -> spatial_bottleneck_dim) -> ConvTranspose2d(spatial_bottleneck_dim -> dim)
        self.spatial_conv = nn.Conv2d(dim, spatial_bottleneck_dim, kernel_size=spatial_kernel_size, padding=padding)
        self.spatial_deconv = nn.ConvTranspose2d(spatial_bottleneck_dim, dim, kernel_size=spatial_kernel_size, padding=padding)

        # --- Final LayerNorm (Eq. 4) --- #
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass for the DFFR module.
        Input shape: (B, C, H, W)
        The data flow remains compatible with the refactored lightweight layers.
        """
        B, C, H, W = x.shape
        assert C == self.dim, "Input channel dimension must match module dimension."
        residual = x

        # --- 1. Dual Pooling & 2. Channel Attention (Eq. 1, 2) --- #
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(B, C)
        x_cat = torch.cat([avg_pool, max_pool], dim=1) # Shape: (B, 2*C)
        
        # 通过瓶颈 MLP 计算通道权重
        channel_weights = self.channel_mlp_w2(F.relu(self.channel_mlp_w1(x_cat)))
        channel_weights = torch.sigmoid(channel_weights).view(B, C, 1, 1)
        
        x_attn = x * channel_weights

        # --- 3. Spatial Enhancement (Eq. 3) --- #
        # 通过瓶颈卷积路径增强空间特征
        x_sp = F.relu(self.spatial_deconv(F.relu(self.spatial_conv(x_attn))))

        # --- 4. Residual Connection & 5. Normalization (Eq. 4) --- #
        refined_x = residual + x_sp

        refined_x = refined_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        refined_x = self.norm(refined_x)
        refined_x = refined_x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return refined_x
