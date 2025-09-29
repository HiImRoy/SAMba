# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

class BottConv(nn.Module):
    """瓶颈卷积 (Bottleneck Convolution)。
    
    这是一个经典的 MobileNet 风格的瓶颈块，包含 1x1 降维、深度可分离卷积和 1x1 升维。
    """
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        # 1x1 卷积: 降维 (瓶颈)
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        # 1x1 卷积: 升维
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

def get_norm_layer(norm_type, channels, num_groups):
    """根据类型获取归一化层。"""
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        # 注意：原始代码为 InstanceNorm3d，对于2D图像通常应为 InstanceNorm2d
        return nn.InstanceNorm2d(channels)

class GBC(nn.Module):
    """全局瓶颈卷积块 (Global Bottleneck Convolution) 的原始实现。"""
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()

        # 定义四个不同的处理块
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x # 保存残差连接

        # 复杂的特征交互路径
        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2 # 两个分支的特征进行逐元素相乘
        x = self.block4(x)

        return x + residual # 添加残差连接
