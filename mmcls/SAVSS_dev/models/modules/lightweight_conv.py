'''
Author: Roy
'''

import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块。

    将标准卷积分解为一个深度卷积 (Depthwise Convolution) 和一个逐点卷积 (Pointwise Convolution)，
    从而在保持感受野的同时，大幅减少参数量和计算量。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        """
        初始化深度可分离卷积。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int): 步长。
            padding (int): 填充大小。
            bias (bool): 是否使用偏置。
        """
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # 深度卷积的输出通道数等于输入通道数
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 分组数等于输入通道数，实现深度卷积
            bias=bias
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 逐点卷积的核大小为 1x1
            bias=bias
        )

    def forward(self, x):
        """
        前向传播。
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
