# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 使用从项目根目录开始的绝对导入路径
from mmcls.models.backbones.samba_crack_encoder import SAMbaCrackEncoder
from mmcls.models.necks.af_neck import AFNeck
from models.MFS import MFSHead


def edge_conv2d(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    使用固定的 Sobel 算子对图像进行边缘检测。

    该函数包含4个方向 (0°, 45°, 90°, 135°) 的 Sobel 卷积核，
    对输入的图像张量进行卷积，并将结果融合成一个边缘图。
    整个过程不可训练。

    Args:
        image_tensor (torch.Tensor): 输入的图像张量，形状为 (B, 3, H, W)。

    Returns:
        torch.Tensor: 输出的边缘图，形状为 (B, 3, H, W)。
    """
    with torch.no_grad():
        image_gray = image_tensor.mean(dim=1, keepdim=True)

        k0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        k45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32)
        k90 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        k135 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32)
        
        kernels = [k0, k45, k90, k135]
        all_edges = []

        for kernel in kernels:
            conv_weight = torch.from_numpy(kernel).view(1, 1, 3, 3).to(image_tensor.device)
            edge = F.conv2d(image_gray, conv_weight, padding=1)
            all_edges.append(edge)

        edge_sum_sq = sum(e**2 for e in all_edges)
        final_edge = torch.sqrt(edge_sum_sq)

        return final_edge.repeat(1, 3, 1, 1)


class SAMbaCrack(nn.Module):
    """
    最终的 SAMbaCrack-AF 模型。

    该类将模块化的 主干(Backbone), 颈部(Neck), 和 头部(Head) 组装成一个
    单一的、端到端的 nn.Module，以便于直接在 Python 脚本中实例化和调用。
    作者: Roy
    """

    def __init__(self, args, **kwargs):
        """
        初始化 SAMbaCrack-AF 模型。

        Args:
            args: 包含所有模型超参数的参数对象。
        """
        super(SAMbaCrack, self).__init__()
        self.args = args

        self.backbone = SAMbaCrackEncoder(
            sam_dims=[96, 192, 384, 768],
            token_dim=256,
            patch_size=8,
            load_height=args.load_height,
            load_width=args.load_width
        )

        self.neck = AFNeck(
            in_channels_list=[96, 192, 384, 768]
        )

        self.head = MFSHead(
            in_channels=[96, 192, 384, 768],
            embedding_dim=8,
            dropout_ratio=0.1
        )

    def forward(self, x):
        """
        定义模型的完整前向传播路径。

        Args:
            x (torch.Tensor): 输入的图像张量，形状为 (B, 3, H, W)。

        Returns:
            torch.Tensor: 模型输出的全分辨率 logits，形状为 (B, 1, H, W)。
        """
        # 1. 生成边缘先验图
        x_edge = edge_conv2d(x)

        # 2. 通过主干网络，将原始图像和边缘图分别送入两个分支
        #    - x (原始图像) -> SAM/Hiera 分支
        #    - x_edge (边缘图) -> SAVSS/Mamba 分支
        feats_sam, feats_mamba = self.backbone(x, x_edge)

        # 3. 通过颈部模块，得到一个融合后的特征金字塔
        fused_feats = self.neck(feats_sam, feats_mamba)

        # 4. 通过解码器头，直接得到最终的全分辨率 logits
        logits = self.head(fused_feats)

        return logits

    def init_weights(self, pretrained=None):
        """
        初始化权重。这里主要用于加载 SAM 编码器的预训练权重。
        """
        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights(pretrained)
