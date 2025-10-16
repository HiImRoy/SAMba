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
from models.PatchMerging import PatchMerging

from mmcls.SAVSS_dev.models.samba_unet_modules.hiera import Hiera
from mmcls.SAVSS_dev.models.samba_unet_modules.refiner_adapter import DynamicFeatureFusionRefiner, MLPAdapter
from mmcls.SAVSS_dev.models.samba_unet_modules.hoacm import HOACM
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer
from models.decoder import SAVSS_UNet_Decoder

logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class SAMbaCrack(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.neck_dims = [96, 192, 384, 768]

        # --- Hiera (SAM) 分支参数 ---
        hiera_depths = getattr(args, 'hiera_depths', (2, 2, 6, 2))
        hiera_num_heads = getattr(args, 'hiera_num_heads', (2, 4, 8, 16))

        # --- SAVSS 分支参数 ---
        savss_depths = [2, 2, 2, 2]
        savss_drop_path_rate = getattr(args, 'savss_drop_path_rate', 0.1)
        savss_use_rms_norm = getattr(args, 'savss_use_rms_norm', True)
        savss_with_dwconv = getattr(args, 'savss_with_dwconv', True)

        # --- 1. Hiera (SAM) 分支 --- #
        self.sam_encoder = Hiera(
            img_size=args.load_height,
            patch_size=4,
            embed_dim=self.neck_dims[0],
            depths=hiera_depths,
            num_heads=hiera_num_heads,
            out_indices=(0, 1, 2, 3)
        )
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # --- Adapter and Refiner for SAM features --- #
        self.refiners = nn.ModuleList([DynamicFeatureFusionRefiner(dim=d) for d in self.neck_dims])
        self.adapters = nn.ModuleList([MLPAdapter(dim=d) for d in self.neck_dims])

        # --- 2. SAVSS 分层编码器 --- #
        self.savss_patch_embed = PatchEmbed(
            img_size=args.load_height,
            in_channels=3,
            embed_dims=self.neck_dims[0],
            conv_cfg={"kernel_size": 4, "stride": 4}
        )
        num_patches = self.savss_patch_embed.num_patches
        self.savss_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.neck_dims[0]))
        nn.init.trunc_normal_(self.savss_pos_embed, std=0.02)

        self.savss_stages = nn.ModuleList()
        self.savss_downsamplers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, savss_drop_path_rate, sum(savss_depths))]
        layer_idx_offset = 0

        for i in range(4):
            dim = self.neck_dims[i]
            depth = savss_depths[i]

            stage_layers = nn.ModuleList()
            for j in range(depth):
                mamba_cfg = {'d_state': 16, 'expand': 2, 'dt_rank': 'auto', 'conv_size': 7}
                stage_layers.append(SAVSS_Layer(
                    embed_dims=dim,
                    use_rms_norm=savss_use_rms_norm,
                    with_dwconv=savss_with_dwconv,
                    drop_path_rate=dpr[layer_idx_offset + j],
                    mamba_cfg=mamba_cfg
                ))
            self.savss_stages.append(stage_layers)
            layer_idx_offset += depth

            if i < 3:
                self.savss_downsamplers.append(
                    PatchMerging(dim=self.neck_dims[i], norm_layer=nn.LayerNorm)
                )

        # --- 3. 融合与解码器模块 --- #
        self.hoacms = nn.ModuleList([HOACM(dim=d) for d in self.neck_dims])

        decoder_savss_args = dict(
            use_rms_norm=savss_use_rms_norm,
            with_dwconv=savss_with_dwconv,
            drop_path_rate=0.0,
            mamba_cfg={'d_state': 16, 'expand': 2, 'dt_rank': 'auto', 'conv_size': 7}
        )

        self.decoder = SAVSS_UNet_Decoder(
            decoder_dims=list(reversed(self.neck_dims)),
            savss_layer_args=decoder_savss_args
        )

    def forward_savss_encoder(self, x):
        B = x.shape[0]
        # [Roy-修正] 遵循标准实践，先获取PatchEmbed的输出，再手动计算H和W
        x = self.savss_patch_embed(x)
        _B, L, _C = x.shape
        H = W = int(L**0.5)

        x = x + self.savss_pos_embed
        mamba_features = []
        for i in range(4):
            for layer in self.savss_stages[i]:
                x = layer(x, hw_shape=(H, W))
            C = self.neck_dims[i]
            mamba_features.append(x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous())
            if i < 3:
                x, H, W = self.savss_downsamplers[i](x, H, W)
        return mamba_features

    def forward(self, x, stage=3):
        if stage == 1:
            mamba_features = self.forward_savss_encoder(x)
            decoder_input = (mamba_features[3], mamba_features[2], mamba_features[1], mamba_features[0])
            logits = self.decoder(decoder_input)
            return logits

        mamba_features = self.forward_savss_encoder(x)
        with torch.no_grad():
            sam_native_features = self.sam_encoder(x)

        sam_features = []
        for i in range(len(sam_native_features)):
            adapted = self.adapters[i](sam_native_features[i])
            refined = self.refiners[i](sam_native_features[i])
            sam_features.append(adapted + refined)

        fused_features = []
        for i in range(4):
            sam_feat = sam_features[i]
            mamba_feat = mamba_features[i]
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
