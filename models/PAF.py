# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GBC import BottConv

class PAF(nn.Module):
    """金字塔注意力融合模块 (Pyramid Attention Fusion) 的原始实现。 
    
    该模块通过计算一个相似度图 (similarity map) 来动态地融合
    一个基础特征 (base_feat) 和一个引导特征 (guidance_feat)。
    """
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 after_relu: bool = False,
                 mid_norm: nn.Module = nn.BatchNorm2d,
                 in_norm: nn.Module = nn.BatchNorm2d):
        super().__init__()
        self.after_relu = after_relu

        # 用于将输入特征转换到中间维度的瓶颈卷积
        self.feature_transform = nn.Sequential(
            BottConv(in_channels, mid_channels, mid_channels=16, kernel_size=1),
            mid_norm(mid_channels)
        )

        # 用于将融合后的特征转换回输入维度的通道适配器
        self.channel_adapter = nn.Sequential(
            BottConv(mid_channels, in_channels, mid_channels=16, kernel_size=1),
            in_norm(in_channels)
        )

        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, base_feat: torch.Tensor, guidance_feat: torch.Tensor) -> torch.Tensor:
        base_shape = base_feat.size()

        if self.after_relu:
            base_feat = self.relu(base_feat)
            guidance_feat = self.relu(guidance_feat)

        # 分别转换引导特征和基础特征
        guidance_query = self.feature_transform(guidance_feat)
        base_key = self.feature_transform(base_feat)
        
        # 将引导特征的尺寸插值到与基础特征一致
        guidance_query = F.interpolate(guidance_query, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)
        
        # 通过将转换后的特征相乘，并经过通道适配器和 sigmoid 来计算相似度图
        similarity_map = torch.sigmoid(self.channel_adapter(base_key * guidance_query))
        
        # 将原始引导特征的尺寸也插值到与基础特征一致
        resized_guidance = F.interpolate(guidance_feat, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)

        # 使用相似度图作为权重，在基础特征和引导特征之间进行加权融合
        fused_feature = (1 - similarity_map) * base_feat + similarity_map * resized_guidance

        return fused_feature
