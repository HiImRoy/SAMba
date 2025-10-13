# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用从项目根目录开始的绝对导入路径
from mmcls.models.backbones.samba_crack_encoder import SAMbaCrackEncoder
from mmcls.models.necks.af_neck import AFNeck
from mmcls.models.heads.mfs_head import MFSHead


class SAMbaCrackAF(nn.Module):
    """
    最终的 SAMbaCrack-AF 模型。

    该类将模块化的 主干(Backbone), 颈部(Neck), 和 头部(Head) 组装成一个
    单一的、端到端的 nn.Module，以便于直接在 Python 脚本中实例化和调用。
    """

    def __init__(self, args, **kwargs):
        """
        初始化 SAMbaCrack-AF 模型。

        Args:
            args: 包含所有模型超参数的参数对象。
        """
        super(SAMbaCrackAF, self).__init__()
        self.args = args

        # 1. 实例化主干网络 (Backbone)
        self.backbone = SAMbaCrackEncoder(
            sam_dims=[96, 192, 384, 768],
            token_dim=256,
            patch_size=8,
            load_height=args.load_height,
            load_width=args.load_width
        )

        # 2. 实例化颈部模块 (Neck)
        self.neck = AFNeck(
            in_channels_list=[96, 192, 384, 768]
        )

        # 3. 实例化解码器头 (Head)
        # [FIXED] 明确使用 embedding_dim=8 来实例化轻量化的全分辨率 MFSHead
        self.head = MFSHead(
            in_channels_list=[96, 192, 384, 768],
            embedding_dim=8, # <--- 已根据您的最终要求进行修改
            num_classes=1
        )

    def forward(self, x):
        """
        定义模型的完整前向传播路径。

        Args:
            x (torch.Tensor): 输入的图像张量，形状为 (B, 3, H, W)。

        Returns:
            torch.Tensor: 模型输出的全分辨率 logits，形状为 (B, 1, H, W)。
        """

        # 1. 通过主干网络，得到两个并行的特征金字塔
        feats_sam, feats_mamba = self.backbone(x)

        # 2. 通过颈部模块，得到一个融合后的特征金字塔
        fused_feats = self.neck(feats_sam, feats_mamba)

        # 3. 通过解码器头，直接得到最终的全分辨率 logits
        logits = self.head(fused_feats)

        # 4. [REMOVED] 不再需要最后的上采样步骤，因为 MFSHead 已输出全分辨率结果
        # logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits

    def init_weights(self, pretrained=None):
        """
        初始化权重。这里主要用于加载 SAM 编码器的预训练权重。
        """
        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights(pretrained)
