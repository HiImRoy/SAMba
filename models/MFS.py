# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.GBC import GBC, BottConv
from models.DySample import DySample

class MLP(nn.Module):
    """简单的多层感知机，用于特征维度映射。"""
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class MFS(nn.Module):
    """多尺度特征分割头 (Multi-scale Feature Segmentation Head)。
    
    该模块作为解码器，接收来自编码器不同层级的特征图，
    并将它们融合以生成最终的分割结果。
    """
    def __init__(self, embedding_dim):  # 在我们的模型中 embedding_dim=8
        super(MFS, self).__init__()

        self.embedding_dim = embedding_dim
        # 定义 MLP 层，将不同通道数的输入特征图统一到 embedding_dim
        # c4: 最高层特征 (来自 SAMbaCrack 的 projections[3]), 通道 128 -> embedding_dim
        # c3: ... (来自 projections[2]), 通道 64 -> embedding_dim
        # c2: ... (来自 projections[1]), 通道 32 -> embedding_dim
        # c1: 最低层特征 (来自 projections[0]), 通道 16 -> embedding_dim
        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16, embed_dim=embedding_dim)
        
        # 用于处理拼接后特征的 GBC 模块
        self.GBC_C = GBC(embedding_dim*4) # 8*4=32
        self.GBC_8 = GBC(8, norm_type='IN')
        self.GN_C = nn.GroupNorm(num_channels=embedding_dim*4, num_groups=embedding_dim*4//16)
        # 用于融合后降维的瓶颈卷积
        self.linear_fuse = BottConv(embedding_dim*4, embedding_dim, embedding_dim//8, kernel_size=1, padding=0, stride=1)

        # 最终的预测层
        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

        # 动态上采样模块，用于统一特征图的空间分辨率
        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)

    def forward(self, inputs):
        # inputs 是一个包含 c4, c3, c2, c1 的元组
        c4, c3, c2, c1 = inputs
        
        # --- 处理 c4 (最高层) ---
        b, c, h, w = c4.shape
        # 维度映射: (B, C, H, W) -> (B, H*W, C) -> (B, H*W, E) -> (B, E, H, W)
        out_c4 = self.linear_c4(c4.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c4 = self.DySample_C_8(out_c4) # 上采样 8 倍

        # --- 处理 c3 ---
        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(c3.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c3 = self.DySample_C_4(out_c3) # 上采样 4 倍

        # --- 处理 c2 ---
        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(c2.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c2 = self.DySample_C_2(out_c2) # 上采样 2 倍

        # --- 处理 c1 (最低层) ---
        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(c1.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)

        # --- 融合 ---
        # 沿通道维度拼接所有处理后的特征图
        out_c = self.GBC_C(torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1))
        # 使用 1x1 卷积进行特征融合和降维
        out_c = self.linear_fuse(out_c)

        # --- 生成预测 ---
        out_c = self.dropout(out_c)
        x = self.linear_pred_1(self.linear_pred(out_c))

        # 将最终输出上采样到与最大特征图 (c1) 相同的尺寸
        _, _, H, W = c1.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x
