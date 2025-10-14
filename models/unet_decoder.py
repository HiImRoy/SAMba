# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 DySample 模块已在此文件中定义
from models.DySample import DySample

# --- 辅助模块: ConvBlock ---
class ConvBlock(nn.Module):
    """标准的卷积块，包含两个 Conv-BN-ReLU 序列。"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# --- 【核心模块】全新的 U-Net 风格解码器头 ---
class UNetDecoderHead(nn.Module):
    """
    U-Net 风格解码器头。

    该模块将来自颈部的融合特征金字塔，通过逐级上采样和跳跃连接的方式，
    进行精细的解码，并采用一种优化的最终输出策略生成全分辨率预测。
    作者: Roy
    """
    def __init__(self, in_channels=[96, 192, 384, 768], final_embedding_dim=16):
        super(UNetDecoderHead, self).__init__()

        # --- 1. U-Net 解码路径 --- #
        # Stage 4 (最底层)
        self.conv4 = ConvBlock(in_channels[3], in_channels[3]) # (B, 768, 14, 14)

        # Stage 3 (上采样与融合)
        self.up3 = nn.ConvTranspose2d(in_channels[3], in_channels[2], kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels[3], in_channels[2]) # 768 (cat) -> 384

        # Stage 2 (上采样与融合)
        self.up2 = nn.ConvTranspose2d(in_channels[2], in_channels[1], kernel_size=2, stride=2)
        self.conv2 = ConvBlock(in_channels[2], in_channels[1]) # 384 (cat) -> 192

        # Stage 1 (上采样与融合)
        self.up1 = nn.ConvTranspose2d(in_channels[1], in_channels[0], kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels[1], in_channels[0]) # 192 (cat) -> 96

        # --- 2. 优化的最终输出策略 --- #
        # a. 通道降维: 将最后一级融合后的特征降维至一个轻量级的中间维度
        self.final_conv_reduce = nn.Conv2d(in_channels[0], final_embedding_dim, kernel_size=1)
        
        # b. 动态上采样: 使用 DySample 将轻量级特征图高效上采样至全分辨率
        self.final_upsample = DySample(final_embedding_dim, scale=4)

        # c. 最终预测: 在全分辨率上进行最终的 1x1 卷积，生成 logits
        self.final_predictor = nn.Conv2d(final_embedding_dim, 1, kernel_size=1)


    def forward(self, inputs):
        """
        前向传播函数。
        Args:
            inputs (list/tuple): 包含4个融合特征图 [c1, c2, c3, c4] 的列表/元组。
        """
        c1, c2, c3, c4 = inputs

        # --- U-Net 解码路径 --- #
        # Stage 4: (B, 768, 14, 14) -> (B, 768, 14, 14)
        d4 = self.conv4(c4)

        # Stage 3: (B, 768, 14, 14) -> (B, 384, 28, 28)
        d3_up = self.up3(d4) # (B, 384, 28, 28)
        d3_cat = torch.cat([d3_up, c3], dim=1) # (B, 768, 28, 28)
        d3 = self.conv3(d3_cat)

        # Stage 2: (B, 384, 28, 28) -> (B, 192, 56, 56)
        d2_up = self.up2(d3) # (B, 192, 56, 56)
        d2_cat = torch.cat([d2_up, c2], dim=1) # (B, 384, 56, 56)
        d2 = self.conv2(d2_cat)

        # Stage 1: (B, 192, 56, 56) -> (B, 96, 112, 112)
        d1_up = self.up1(d2) # (B, 96, 112, 112)
        d1_cat = torch.cat([d1_up, c1], dim=1) # (B, 192, 112, 112)
        d1 = self.conv1(d1_cat)

        # --- 优化的最终输出策略 --- #
        # a. 通道降维: (B, 96, 112, 112) -> (B, 16, 112, 112)
        d_final_reduced = self.final_conv_reduce(d1)

        # b. 动态上采样: (B, 16, 112, 112) -> (B, 16, 448, 448)
        d_final_upsampled = self.final_upsample(d_final_reduced)

        # c. 最终预测: (B, 16, 448, 448) -> (B, 1, 448, 448)
        logits = self.final_predictor(d_final_upsampled)

        return logits
