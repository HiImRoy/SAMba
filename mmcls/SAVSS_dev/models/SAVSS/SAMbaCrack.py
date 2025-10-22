# Author: Roy
# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# **REFACTOR**: Replaced the SAVSS Mamba encoder with the new MambaVision backbone.
# **REFACTOR**: The model now dynamically calculates Mamba feature dimensions based on the variant.
# **FEATURE**: Added `mamba_variant` and pretrained weight loading for the new backbone.
# REFACTOR: Removed savss_depths, MambaVision now controls its own depths based on variant.
# FIX: Updated mamba_dims to reflect that the final stage of the backbone now downsamples.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

# --- 从项目根目录进行正确的绝对导入 ---
from mmcls.SAVSS_dev.models.samba_unet_modules.hiera import Hiera
from models.fcm import FCM
# --- [修改] 导入新的 MambaVision 主干网络 --- 
from models.mambavision_backbone import create_mamba_vision_backbone


# --- 【重构】使用 Sequential Adapter 进行微调 ---
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.norm1.normalized_shape[0]
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped_x = x + prompt
        net = self.block(promped_x)
        return net


# --- U-Net 风格解码器模块 (保持不变) ---
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, decoder_channels):
        super().__init__()
        c1_dim, c2_dim, c3_dim, c4_dim = decoder_channels
        self.block1 = DecoderBlock(in_channels=c4_dim, skip_channels=c3_dim, out_channels=c3_dim)
        self.block2 = DecoderBlock(in_channels=c3_dim, skip_channels=c2_dim, out_channels=c2_dim)
        self.block3 = DecoderBlock(in_channels=c2_dim, skip_channels=c1_dim, out_channels=c1_dim)
        self.segmentation_head = nn.Conv2d(c1_dim, 1, kernel_size=1)

    def forward(self, features, final_size):
        c4, c3, c2, c1 = features
        x = self.block1(c4, c3)
        x = self.block2(x, c2)
        x = self.block3(x, c1)
        logits = self.segmentation_head(x)
        logits = F.interpolate(logits, size=final_size, mode='bilinear', align_corners=False)
        return logits


logger = logging.getLogger(__name__)

class SAMbaCrack(nn.Module):
    """ 最终的 SAMbaCrack 模型 (已更新为使用 MambaVision 主干) """

    def __init__(self, args, **kwargs):
        super().__init__()

        # --- [修改] 模型超参数定义 ---
        # 1. SAM (Hiera) 分支参数
        self.sam_dims = [112, 224, 448, 896]
        self.hiera_depths = getattr(args, 'hiera_depths', (2, 3, 16, 3))
        self.hiera_num_heads = getattr(args, 'hiera_num_heads', (2, 4, 8, 16))
        self.sam_patch_size = 4

        # 2. Mamba (MambaVision) 分支参数
        self.mamba_variant = getattr(args, 'mamba_variant', 'T')
        # REFACTOR: savss_depths is now controlled by MambaVision itself based on variant.
        drop_path_rate = getattr(args, 'savss_drop_path_rate', 0.2) # 继续使用 savss_drop_path_rate
        mamba_pretrained = getattr(args, 'mamba_pretrained', False)
        mamba_pretrained_weights = getattr(args, 'mamba_pretrained_weights', '')

        # 3. Neck (FCM) & Decoder 参数
        self.fcm_dims = [64, 128, 256, 512]

        # --- [新增] 动态计算 MambaVision 的输出维度 ---
        # MambaVision 不同变体的初始维度
        variant_dims = {'T': 64, 'S': 96, 'B': 128, 'L': 196}
        mamba_base_dim = variant_dims[self.mamba_variant.upper()]
        # [FIX] 更新 mamba_dims 以反映现在使用的是下采样前的特征
        # 这些维度是每个 MambaVisionStage 的输入维度
        self.mamba_dims = [
            mamba_base_dim * (2 ** 0),
            mamba_base_dim * (2 ** 1),
            mamba_base_dim * (2 ** 2),
            mamba_base_dim * (2 ** 3)
        ]

        # --- 1. Hiera (SAM) 分支 (保持不变) --- #
        self.sam_encoder = Hiera(
            img_size=args.load_height,
            patch_size=self.sam_patch_size,
            embed_dim=self.sam_dims[0],
            depths=self.hiera_depths,
            num_heads=self.hiera_num_heads,
            out_indices=(0, 1, 2, 3)
        )
        for param in self.sam_encoder.parameters():
            param.requires_grad = False
        for level in self.sam_encoder.levels:
            sequential_blocks = level['blocks']
            for i in range(len(sequential_blocks)):
                sequential_blocks[i] = Adapter(sequential_blocks[i])

        # --- 2. [重构] Mamba (MambaVision) 分支 --- #
        self.mamba_encoder = create_mamba_vision_backbone(
            variant=self.mamba_variant,
            # depths=self.savss_depths, # REFACTOR: Removed, MambaVision now controls its own depths.
            drop_path_rate=drop_path_rate,
            pretrained=mamba_pretrained,
            pretrained_path=mamba_pretrained_weights
        )

        # --- 3. 融合与解码器模块 (颈部) --- #
        self.sam_adapters = nn.ModuleList([
            nn.Conv2d(self.sam_dims[i], self.fcm_dims[i], kernel_size=1) for i in range(4)
        ])
        # [修改] 使用动态计算的 mamba_dims 初始化 adapter
        self.mamba_adapters = nn.ModuleList([
            nn.Conv2d(self.mamba_dims[i], self.fcm_dims[i], kernel_size=1) for i in range(4)
        ])
        self.fcms = nn.ModuleList([FCM(dim=d) for d in self.fcm_dims])
        self.decoder = UNetDecoder(decoder_channels=self.fcm_dims)

    def forward(self, x):
        # 1. Hiera (SAM) 分支特征提取
        sam_features = self.sam_encoder(x)

        # 2. [重构] Mamba (MambaVision) 分支特征提取
        # 新的主干网络直接接收图像输入并返回多尺度特征图元组
        mamba_features = self.mamba_encoder(x)

        # 3. 特征融合
        fused_features = []
        for i in range(4):
            sam_feat = sam_features[i]
            mamba_feat = mamba_features[i]

            # 空间维度对齐 (保持不变)
            if sam_feat.shape[-2:] != mamba_feat.shape[-2:]:
                sam_feat = F.interpolate(
                    sam_feat, 
                    size=mamba_feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 通道维度对齐 (保持不变)
            sam_feat_adapted = self.sam_adapters[i](sam_feat)
            mamba_feat_adapted = self.mamba_adapters[i](mamba_feat)

            # 调用 FCM 模块进行融合
            fused = self.fcms[i](sam_feat_adapted, mamba_feat_adapted)
            fused_features.append(fused)

        # 4. 解码器生成预测
        decoder_input = (fused_features[3], fused_features[2], fused_features[1], fused_features[0])
        logits = self.decoder(decoder_input, final_size=x.shape[-2:])

        return logits

    def init_weights(self, pretrained=None):
        """初始化 Hiera 编码器的预训练权重。MambaVision 的权重在创建时加载。"""
        if pretrained is None:
            logger.info("没有为 Hiera 提供预训练权重。")
            return
        if os.path.isfile(pretrained):
            logger.info(f"从以下位置加载 Hiera 编码器的预训练权重: {pretrained}")
            try:
                state_dict = torch.load(pretrained, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                
                missing_keys, unexpected_keys = self.sam_encoder.load_state_dict(state_dict, strict=False)
                missing_keys_non_adapter = [k for k in missing_keys if 'prompt_learn' not in k]
                
                if not missing_keys_non_adapter and not unexpected_keys:
                    logger.info("Hiera: 所有非 Adapter 参数均已成功加载。")
                else:
                    logger.warning(f"Hiera 权重加载不完全: 缺少 {len(missing_keys_non_adapter)} 个, 意外 {len(unexpected_keys)} 个.")

            except Exception as e:
                logger.error(f"加载 Hiera 预训练权重时出错: {e}")
        else:
            logger.warning(f"在以下位置未找到 Hiera 预训练权重文件: {pretrained}。")
