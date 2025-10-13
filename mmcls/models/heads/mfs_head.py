# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用从项目根目录开始的绝对导入路径
from models.GBC import GBC
from models.DySample import DySample


class MFSHead(nn.Module):
    """
    多尺度特征分割头 (Multi-scale Feature Segmentation Head, MFSHead)。
    【最终轻量化全分辨率版】

    该解码器头严格遵循最终设计蓝图，将融合后的多尺度特征直接解码为
    最终的全分辨率像素级预测，并采用了轻量化的内部处理维度。
    """

    def __init__(self, 
                 in_channels_list,
                 embedding_dim, # <--- 根据您的要求，这个参数现在将是 8
                 num_classes=1, 
                 init_cfg=None):
        """
        初始化 MFSHead (最终版)。

        Args:
            in_channels_list (list[int]): 输入的特征金字塔的通道数列表。
                                          例如: [96, 192, 384, 768]
            embedding_dim (int): 一个统一的内部处理维度。根据您的要求，这里将设为 8。
            num_classes (int): 分割任务的类别数。默认为 1 (二分类)。
            init_cfg (dict, optional): (未使用) 用于初始化的配置字典。
        """
        super(MFSHead, self).__init__()
        self.num_classes = num_classes

        # --- 1. 创建并行的 MLP (1x1 Conv) 层 --- #
        # 用于将不同尺度的输入特征统一投影到 embedding_dim (8)
        self.linear_c1 = nn.Conv2d(in_channels_list[0], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels_list[1], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels_list[2], embedding_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(in_channels_list[3], embedding_dim, kernel_size=1)

        # --- 2. 创建并行上采样层 (上采样至全分辨率) --- #
        # 输入通道数现在是 embedding_dim (8)
        self.upsample_c1 = DySample(in_channels=embedding_dim, scale=4)
        self.upsample_c2 = DySample(in_channels=embedding_dim, scale=8)
        self.upsample_c3 = DySample(in_channels=embedding_dim, scale=16)
        self.upsample_c4 = DySample(in_channels=embedding_dim, scale=32)

        # --- 3. 创建全分辨率深度融合层 --- #
        # 拼接后的总通道数现在是 8 * 4 = 32
        fused_channels = embedding_dim * 4
        self.fusion_gbc = GBC(fused_channels)
        # 使用 1x1 卷积进行通道融合，从 32 降维到 8
        self.fusion_conv = nn.Conv2d(fused_channels, embedding_dim, kernel_size=1)

        # --- 4. 创建全分辨率最终预测层 --- #
        # 将融合后的特征 (8通道) 投影到最终的类别数 (1)
        self.final_conv = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, features):
        """
        前向传播函数。
        """
        c1, c2, c3, c4 = features

        # --- A. 并行通道对齐 ---
        # 输出 c1_p, c2_p, c3_p, c4_p 的通道数都为 8
        c1_p = self.linear_c1(c1)
        c2_p = self.linear_c2(c2)
        c3_p = self.linear_c3(c3)
        c4_p = self.linear_c4(c4)

        # --- B. 并行空间对齐 (上采样至全分辨率) ---
        # 输出 c1_up, c2_up, c3_up, c4_up 的维度都为 (B, 8, 448, 448)
        c1_up = self.upsample_c1(c1_p)
        c2_up = self.upsample_c2(c2_p)
        c3_up = self.upsample_c3(c3_p)
        c4_up = self.upsample_c4(c4_p)

        # --- C. 全分辨率拼接 ---
        # fused 的维度为 (B, 32, 448, 448)
        fused = torch.cat([c1_up, c2_up, c3_up, c4_up], dim=1)

        # --- D. & E. 全分辨率深度融合与预测 ---
        fused = self.fusion_gbc(fused) # (B, 32, 448, 448)
        fused = self.fusion_conv(fused) # (B, 8, 448, 448)
        logits = self.final_conv(fused) # (B, 1, 448, 448)

        return logits
