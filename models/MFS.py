# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.GBC import GBC
from models.DySample import DySample

class MLP(nn.Module):
    """简单的1x1卷积，用于特征维度映射。"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.proj(x)
        return x

class MFSHead(nn.Module):
    """
    全分辨率多尺度特征分割头 (Full-Resolution Multi-scale Feature Segmentation Head).
    该模块将来自颈部的多尺度特征图并行上采样至全分辨率，然后进行深度融合，
    最终直接输出全分辨率的分割预测图。
    """
    def __init__(self, in_channels=[96, 192, 384, 768], embedding_dim=8, dropout_ratio=0.1):
        super(MFSHead, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # A. 并行通道对齐 (MLPs)
        # 使用独立的 1x1 Conv 将不同通道数的输入特征图统一到 embedding_dim
        self.c1_mlp = MLP(input_dim=self.in_channels[0], embed_dim=self.embedding_dim)
        self.c2_mlp = MLP(input_dim=self.in_channels[1], embed_dim=self.embedding_dim)
        self.c3_mlp = MLP(input_dim=self.in_channels[2], embed_dim=self.embedding_dim)
        self.c4_mlp = MLP(input_dim=self.in_channels[3], embed_dim=self.embedding_dim)

        # B. 并行空间对齐 (上采样至全分辨率)
        # c1 (112x112) -> 448x448 (x4)
        # c2 (56x56)   -> 448x448 (x8)
        # c3 (28x28)   -> 448x448 (x16)
        # c4 (14x14)   -> 448x448 (x32)
        self.upsample_c1 = DySample(self.embedding_dim, scale=4)
        self.upsample_c2 = DySample(self.embedding_dim, scale=8)
        self.upsample_c3 = DySample(self.embedding_dim, scale=16)
        self.upsample_c4 = DySample(self.embedding_dim, scale=32)

        # D. 全分辨率深度融合
        self.fusion_gbc = GBC(in_channels=self.embedding_dim * 4)
        self.fusion_conv = nn.Conv2d(self.embedding_dim * 4, self.embedding_dim, kernel_size=1, bias=False)
        
        # E. 全分辨率预测
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0. else nn.Identity()
        self.final_conv = nn.Conv2d(self.embedding_dim, 1, kernel_size=1)


    def forward(self, inputs):
        # inputs: (c1, c2, c3, c4)
        # c1: (B, 96, 112, 112)
        # c2: (B, 192, 56, 56)
        # c3: (B, 384, 28, 28)
        # c4: (B, 768, 14, 14)
        c1, c2, c3, c4 = inputs

        # A. 并行通道对齐
        c1_p = self.c1_mlp(c1)
        c2_p = self.c2_mlp(c2)
        c3_p = self.c3_mlp(c3)
        c4_p = self.c4_mlp(c4)

        # B. 并行空间对齐
        upsample_c1 = self.upsample_c1(c1_p)
        upsample_c2 = self.upsample_c2(c2_p)
        upsample_c3 = self.upsample_c3(c3_p)
        upsample_c4 = self.upsample_c4(c4_p)

        # C. 全分辨率拼接
        fused_feats = torch.cat([upsample_c1, upsample_c2, upsample_c3, upsample_c4], dim=1)

        # D. 全分辨率深度融合
        fused_feats = self.fusion_gbc(fused_feats)
        fused_feats = self.fusion_conv(fused_feats)

        # E. 全分辨率预测
        fused_feats = self.dropout(fused_feats)
        logits = self.final_conv(fused_feats)

        return logits
