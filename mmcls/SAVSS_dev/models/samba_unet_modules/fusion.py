# Author: Roy
# Based on the refactoring request.

import torch
import torch.nn as nn

class AdaptiveFusionModule(nn.Module):
    """轻量级自适应门控融合模块。

    该模块通过学习一个空间注意力门控，动态地、逐像素地融合两个特征图。
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # 初步融合层：将拼接后的特征降维回原始通道数
        self.preliminary_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 门控生成层：从融合特征中学习空间注意力门控
        self.gate_generator = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_sam, x_mamba):
        """
        Args:
            x_sam (torch.Tensor): 来自 SAM 分支的特征图。Shape: (B, C, H, W)
            x_mamba (torch.Tensor): 来自 Mamba 分支的特征图。Shape: (B, C, H, W)

        Returns:
            torch.Tensor: 融合后的特征图。Shape: (B, C, H, W)
        """
        # 步骤 1: 沿通道维度拼接两个输入特征
        # Shape: (B, C, H, W) + (B, C, H, W) -> (B, 2*C, H, W)
        x_concat = torch.cat([x_sam, x_mamba], dim=1)

        # 步骤 2: 初步融合，将通道数降维
        # Shape: (B, 2*C, H, W) -> (B, C, H, W)
        fused_feature = self.preliminary_fusion(x_concat)

        # 步骤 3: 生成单通道的空间门控，并应用 sigmoid
        # Shape: (B, C, H, W) -> (B, 1, H, W)
        gate = torch.sigmoid(self.gate_generator(fused_feature))

        # 步骤 4: 使用门控加权融合原始特征
        # gate 会被自动广播以匹配 x_sam 和 x_mamba 的通道数
        # output = x_sam * gate + x_mamba * (1 - gate)
        output = x_sam * gate + x_mamba * (1 - gate)

        return output
