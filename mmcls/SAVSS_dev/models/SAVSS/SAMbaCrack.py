# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 使用从项目根目录开始的绝对导入路径
from mmcls.models.backbones.samba_crack_encoder import SAMbaCrackEncoder
from mmcls.models.necks.af_neck import AFNeck
# --- [MODIFIED] 导入全新的 UNetDecoderHead --- #
from models.unet_decoder import UNetDecoderHead


def edge_conv2d(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    使用固定的 Sobel 算子对图像进行边缘检测。
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
    """

    def __init__(self, args, **kwargs):
        """
        初始化 SAMbaCrack-AF 模型。
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

        # --- [MODIFIED] 使用全新的 UNetDecoderHead 替换 MFSHead ---
        self.head = UNetDecoderHead(
            in_channels=[96, 192, 384, 768]
            # 使用 UNetDecoderHead 的默认 final_embedding_dim=16
        )

    def forward(self, x):
        """
        定义模型的完整前向传播路径。
        """
        # 1. 生成边缘先验图 (当前版本已移除，直接使用原始图像)
        # x_edge = edge_conv2d(x)

        # 2. 通过主干网络
        feats_sam, feats_mamba = self.backbone(x, x)

        # 3. 通过颈部模块，得到一个融合后的特征金字塔
        fused_feats = self.neck(feats_sam, feats_mamba)

        # 4. 通过解码器头，直接得到最终的全分辨率 logits
        logits = self.head(fused_feats)

        return logits

    def init_weights(self, pretrained=None):
        """
        初始化权重。
        """
        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights(pretrained)
