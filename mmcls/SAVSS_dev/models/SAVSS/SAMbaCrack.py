# Author: Roy
# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

from mmcls.models.builder import BACKBONES
from mmcls.models.utils.embed import PatchEmbed

from mmcls.SAVSS_dev.models.samba_unet_modules.hiera import Hiera
from mmcls.SAVSS_dev.models.samba_unet_modules.refiner_adapter import DynamicFeatureFusionRefiner, MLPAdapter
from mmcls.SAVSS_dev.models.samba_unet_modules.hoacm import HOACM
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer
from models.DySample import DySample

# --- U-Net 风格解码器模块 (保持不变) ---
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, decoder_channels):
        super().__init__()
        c1_dim, c2_dim, c3_dim, c4_dim = decoder_channels
        self.block1 = DecoderBlock(in_channels=c4_dim, skip_channels=c3_dim, out_channels=c3_dim)
        self.block2 = DecoderBlock(in_channels=c3_dim, skip_channels=c2_dim, out_channels=c2_dim)
        self.block3 = DecoderBlock(in_channels=c2_dim, skip_channels=c1_dim, out_channels=c1_dim)

        # [MODIFIED] Final block to reduce channels before upsampling
        final_out_channels = 16
        self.final_block = nn.Sequential(
            # Reduce channels from c1_dim (96) to 16
            nn.Conv2d(c1_dim, final_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_out_channels),
            nn.ReLU(inplace=True),
            # Upsample to original resolution
            DySample(final_out_channels, scale=2, style='lp', groups=final_out_channels), # 112 -> 224
            DySample(final_out_channels, scale=2, style='lp', groups=final_out_channels), # 224 -> 448
            # Final convolution to get logits
            nn.Conv2d(final_out_channels, 1, kernel_size=1)
        )

    def forward(self, features):
        c4, c3, c2, c1 = features
        x = self.block1(c4, c3)
        x = self.block2(x, c2)
        x = self.block3(x, c1)
        logits = self.final_block(x)
        return logits

logger = logging.getLogger(__name__)

# [NEW] Encapsulate SAVSS stages into a separate module for better profiling
class SAVSSBackbone(nn.Module):
    def __init__(self, backbone_dim, savss_depths, savss_drop_path_rate, savss_use_rms_norm, savss_with_dwconv):
        super().__init__()
        self.savss_stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, savss_drop_path_rate, sum(savss_depths))]
        layer_idx = 0
        for i in range(len(savss_depths)): # Loop through 4 stages
            stage_layers = nn.ModuleList()
            for _ in range(savss_depths[i]): # Loop through 2 layers per stage
                mamba_cfg = {'d_state': 16, 'expand': 2, 'dt_rank': 'auto', 'conv_size': 7}
                stage_layers.append(SAVSS_Layer(
                    embed_dims=backbone_dim,
                    use_rms_norm=savss_use_rms_norm,
                    with_dwconv=savss_with_dwconv,
                    drop_path_rate=dpr[layer_idx],
                    mamba_cfg=mamba_cfg
                ))
                layer_idx += 1
            self.savss_stages.append(stage_layers)

    def forward(self, x_token, hw_shape):
        backbone_taps = []
        for stage in self.savss_stages:
            for layer in stage:
                x_token = layer(x_token, hw_shape)
            backbone_taps.append(x_token)
        return backbone_taps


@BACKBONES.register_module()
class SAMbaCrack(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        # --- [REVISED] 统一的维度配置 --- #
        neck_dims = [96, 192, 384, 768]
        
        # --- Hiera (SAM) 分支参数 ---
        hiera_depths = getattr(args, 'hiera_depths', (2, 2, 6, 2))
        hiera_num_heads = getattr(args, 'hiera_num_heads', (3, 6, 12, 24))

        # --- SAVSS 分支参数 ---
        savss_depths = [2, 2, 2, 2] # 4 stages, 2 layers each
        backbone_dim = 256
        savss_drop_path_rate = getattr(args, 'savss_drop_path_rate', 0.1)
        savss_use_rms_norm = getattr(args, 'savss_use_rms_norm', True)
        savss_with_dwconv = getattr(args, 'savss_with_dwconv', True)

        # --- 1. Hiera (SAM) 分支 --- #
        self.sam_encoder = Hiera(
            img_size=args.load_height,
            patch_size=4,
            embed_dim=neck_dims[0],
            depths=hiera_depths,
            num_heads=hiera_num_heads,
            out_indices=(0, 1, 2, 3)
        )
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # --- [RE-INTRODUCED] Adapter and Refiner for SAM features --- #
        self.refiners = nn.ModuleList([DynamicFeatureFusionRefiner(dim=d) for d in neck_dims])
        self.adapters = nn.ModuleList([MLPAdapter(dim=d) for d in neck_dims])

        # --- 2. SAVSS 单尺度骨干网络 --- #
        self.savss_patch_embed = PatchEmbed(
            img_size=args.load_height,
            in_channels=3, 
            embed_dims=backbone_dim, 
            conv_cfg={"kernel_size": 8, "stride": 8}
        )
        num_patches = (args.load_height // 8) ** 2
        self.savss_pos_embed = nn.Parameter(torch.zeros(1, num_patches, backbone_dim))
        nn.init.trunc_normal_(self.savss_pos_embed, std=0.02)

        # [MODIFIED] Instantiate the new SAVSSBackbone module
        self.savss_backbone_module = SAVSSBackbone(
            backbone_dim=backbone_dim,
            savss_depths=savss_depths,
            savss_drop_path_rate=savss_drop_path_rate,
            savss_use_rms_norm=savss_use_rms_norm,
            savss_with_dwconv=savss_with_dwconv
        )

        # --- 3. SAVSS Neck (对齐到 Hiera 原生维度) --- #
        self.savss_neck = nn.ModuleList([
            # Proj 1: (256, 56, 56) -> (96, 112, 112)
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(backbone_dim, neck_dims[0], kernel_size=1, bias=False),
                nn.BatchNorm2d(neck_dims[0])
            ),
            # Proj 2: (256, 56, 56) -> (192, 56, 56)
            nn.Sequential(
                nn.Conv2d(backbone_dim, neck_dims[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(neck_dims[1])
            ),
            # Proj 3: (256, 56, 56) -> (384, 28, 28)
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(backbone_dim, neck_dims[2], kernel_size=1, bias=False),
                nn.BatchNorm2d(neck_dims[2])
            ),
            # Proj 4: (256, 56, 56) -> (768, 14, 14)
            nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
                nn.Conv2d(backbone_dim, neck_dims[3], kernel_size=1, bias=False),
                nn.BatchNorm2d(neck_dims[3])
            )
        ])

        # --- 4. 融合与解码器模块 --- #
        self.hoacms = nn.ModuleList([HOACM(dim=d) for d in neck_dims])
        self.decoder = UNetDecoder(decoder_channels=neck_dims)

    def forward_savss_encoder(self, x):
        x_token = self.savss_patch_embed(x)
        x_token = x_token + self.savss_pos_embed
        B, L, C = x_token.shape
        H, W = int(L**0.5), int(L**0.5)

        # [MODIFIED] Call the new SAVSSBackbone module
        backbone_taps = self.savss_backbone_module(x_token, (H, W))
        
        taps_2d = [tap.transpose(1, 2).reshape(B, C, H, W) for tap in backbone_taps]

        mamba_features = [
            self.savss_neck[0](taps_2d[0]),
            self.savss_neck[1](taps_2d[1]),
            self.savss_neck[2](taps_2d[2]),
            self.savss_neck[3](taps_2d[3]),
        ]
        return mamba_features

    def forward(self, x, stage=3):
        # Stage 1: Only the SAVSS branch runs
        if stage == 1:
            mamba_features = self.forward_savss_encoder(x)
            decoder_input = (mamba_features[3], mamba_features[2], mamba_features[1], mamba_features[0])
            logits = self.decoder(decoder_input)
            return logits

        # Stage 2 & 3: Both branches run for fusion
        mamba_features = self.forward_savss_encoder(x)
        with torch.no_grad():
            sam_native_features = self.sam_encoder(x)

        # Adapt SAM features before fusion
        sam_features = []
        for i in range(len(sam_native_features)):
            adapted = self.adapters[i](sam_native_features[i])
            refined = self.refiners[i](sam_native_features[i])
            sam_features.append(adapted + refined)

        # Fuse the aligned features from both pyramids
        fused_features = []
        for i in range(4):
            sam_feat = sam_features[i]
            mamba_feat = mamba_features[i]

            # [FIX] Defensive check to ensure spatial dimensions match before fusion
            if sam_feat.shape[-2:] != mamba_feat.shape[-2:]:
                mamba_feat = F.interpolate(
                    mamba_feat, 
                    size=sam_feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            fused = self.hoacms[i](sam_feat, mamba_feat)
            fused_features.append(fused)
        
        decoder_input = (fused_features[3], fused_features[2], fused_features[1], fused_features[0])
        logits = self.decoder(decoder_input)
        return logits

    def init_weights(self, pretrained=None):
        if pretrained is None:
            logger.info("没有提供预训练权重。从头开始初始化 Hiera。")
            return
        if os.path.isfile(pretrained):
            logger.info(f"从以下位置加载 Hiera 编码器的预训练权重: {pretrained}")
            try:
                state_dict = torch.load(pretrained, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                missing_keys, unexpected_keys = self.sam_encoder.load_state_dict(state_dict, strict=False)
                logger.info(f"Hiera 权重加载完毕. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}.")
            except Exception as e:
                logger.error(f"加载预训练权重时出错: {e}")
        else:
            logger.warning(f"在以下位置未找到预训练权重文件: {pretrained}。Hiera 将从头开始初始化。")
