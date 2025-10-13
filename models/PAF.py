# Copyright (c) Roy. All rights reserved.

import torch.nn as nn

class PAF(nn.Module):
    """金字塔注意力融合模块 (Pyramid Attention Fusion)。
    
    该模块通过一个注意力机制，使用一个特征图 (x_s, a.k.a. query) 来
    动态地加权另一个特征图 (x_v, a.k.a. value)，从而实现特征的融合。
    """
    def __init__(self, in_channels, hidden_dim, kernel_size=3, padding=1, stride=1):
        super(PAF, self).__init__()
        # 1x1 卷积，用于将 query 特征投影到隐藏维度
        self.conv_s = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        # 1x1 卷积，用于将 value 特征投影到隐藏维度
        self.conv_v = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 深度可分离卷积，用于高效地生成空间注意力图
        self.conv_spatial = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim)
        
        # 1x1 卷积，用于将融合后的特征投影回原始通道数
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x_v, x_s):
        """
        Args:
            x_v (torch.Tensor): Value 特征图，将被加权。
            x_s (torch.Tensor): Query 特征图，用于生成权重。
        """
        # 将两个输入都投影到相同的隐藏维度
        x_s_proj = self.conv_s(x_s)
        x_v_proj = self.conv_v(x_v)
        
        # 生成空间注意力图
        x_s_proj = self.gelu(x_s_proj)
        attn = self.conv_spatial(x_s_proj)
        attn = self.sigmoid(attn) # 使用 sigmoid 将权重缩放到 (0, 1) 范围
        
        # 在隐藏维度上进行加权融合
        x_att = x_v_proj * attn
        
        # 投影回原始输入维度
        return self.conv_out(x_att)
