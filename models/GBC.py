# Copyright (c) Roy. All rights reserved.

import torch.nn as nn

class BottConv(nn.Module):
    """瓶颈卷积块 (Bottleneck Convolution Block)。
    
    一个高效的卷积模块，通过 1x1 -> 3x3 -> 1x1 的卷积序列来替代单一的
    大核卷积，以减少参数量和计算成本，同时保持相似的感受野。
    """
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, padding, stride):
        super(BottConv, self).__init__()
        self.bott_conv = nn.Sequential(
            # 1x1 卷积，用于降维 (squeeze)
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3x3 卷积，用于特征提取
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1x1 卷积，用于升维 (expand)
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bott_conv(x)

class GBC(nn.Module):
    """全局瓶颈卷积块 (Global Bottleneck Convolution Block)。
    
    在 BottConv 的基础上增加了残差连接，形成一个更强大的特征提取单元。
    残差连接要求输入和输出的通道数必须相同。
    """
    def __init__(self, in_channels, out_channels=None, intermediate_channels=None, norm_type=None):
        super(GBC, self).__init__()
        self.in_channels = in_channels
        # 如果未指定输出通道，则默认为输入通道，以满足残差连接的要求
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.intermediate_channels = intermediate_channels if intermediate_channels is not None else in_channels // 4

        # 核心卷积组件
        self.gwc = BottConv(self.in_channels, self.out_channels, self.intermediate_channels, 3, 1, 1)
        
        # 归一化层
        if norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(self.out_channels)
        else:
            self.norm = nn.BatchNorm2d(self.out_channels)
        
        # 激活函数
        self.act = nn.GELU()

    def forward(self, x):
        # 保存输入用于残差连接
        res = x
        # 通过瓶颈卷积
        x = self.gwc(x)
        # 归一化和激活
        x = self.act(self.norm(x))
        # 添加残差连接
        return x + res
