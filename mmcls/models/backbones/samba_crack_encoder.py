# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用从项目根目录开始的、经过验证的、正确的绝对导入路径和正确的类名
from mmcls.SAVSS_dev.models.samba_unet_modules.hiera import Hiera
from mmcls.SAVSS_dev.models.samba_unet_modules.refiner_adapter import DynamicFeatureFusionRefiner, MLPAdapter
# [MODIFIED] 导入 SAVSS_Block 替换 SAVSS_Layer
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Block
from models.GBC import BottConv
from mmcls.SAVSS_dev.models.modules.patch_embed import ConvPatchEmbed


class SAMbaCrackEncoder(nn.Module):
    """
    SAMbaCrack 模型的双分支主干网络。
    该模块已被重构，以支持接收两个独立的输入，分别送入 SAM 和 Mamba 分支。
    作者: Roy
    """

    def __init__(self, 
                 sam_dims=[96, 192, 384, 768],
                 token_dim=256,
                 patch_size=8, # SAVSS 分支的 patch size
                 load_height=448,
                 load_width=448,
                 hiera_depths=(2, 2, 6, 2),
                 hiera_num_heads=(3, 6, 12, 24),
                 savss_drop_path_rate=0.1,
                 # [REMOVED] 不再需要的参数
                 # savss_use_rms_norm=True,
                 # savss_with_dwconv=True,
                 init_cfg=None):
        super(SAMbaCrackEncoder, self).__init__()

        # --- 1. Hiera (SAM) 分支组件 --- #
        sam_patch_size = 4
        self.sam_patch_embed = ConvPatchEmbed(
            in_channels=3,
            embed_dims=sam_dims[0],
            patch_size=sam_patch_size,
            stride=sam_patch_size
        )
        sam_num_patches = (load_height // sam_patch_size) ** 2
        self.sam_pos_embed = nn.Parameter(torch.zeros(1, sam_num_patches, sam_dims[0]))
        nn.init.trunc_normal_(self.sam_pos_embed, std=0.02)

        self.sam_encoder = Hiera(
            embed_dim=sam_dims[0],
            depths=hiera_depths,
            num_heads=hiera_num_heads,
            out_indices=(0, 1, 2, 3)
        )
        self.refiners = nn.ModuleList([DynamicFeatureFusionRefiner(dim=d) for d in sam_dims])
        self.adapters = nn.ModuleList([MLPAdapter(dim=d) for d in sam_dims])

        # 冻结 SAM 分支的参数
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # --- 2. SAVSS (Mamba) 分支组件 --- #
        self.savss_patch_embed = ConvPatchEmbed(
            in_channels=3, 
            embed_dims=token_dim, 
            patch_size=patch_size,
            stride=patch_size
        )
        savss_num_patches = (load_height // patch_size) ** 2
        self.savss_pos_embed = nn.Parameter(torch.zeros(1, savss_num_patches, token_dim))
        nn.init.trunc_normal_(self.savss_pos_embed, std=0.02)

        self.mamba_encoder = nn.ModuleList()
        # 使用 sum(hiera_depths) 可能不准确，因为Mamba分支的深度是固定的4
        dpr = [x.item() for x in torch.linspace(0, savss_drop_path_rate, 4)]

        for i in range(4):
            mamba_cfg = {'d_state': 16, 'expand': 2, 'dt_rank': 'auto', 'conv_size': 7}
            # [MODIFIED] 实例化 SAVSS_Block
            self.mamba_encoder.append(SAVSS_Block(
                embed_dims=token_dim,
                mamba_cfg=mamba_cfg,
                drop_path_rate=dpr[i]
                # 使用 SAVSS_Block 的默认参数 num_gbc=2
            ))

        self.savss_target_sizes = [
            (load_height // 4, load_width // 4),   # 112x112
            (load_height // 8, load_width // 8),   # 56x56
            (load_height // 16, load_width // 16), # 28x28
            (load_height // 32, load_width // 32), # 14x14
        ]
        self.savss_decoder_heads = nn.ModuleList()
        for i in range(4):
            decoder_head = nn.Sequential(
                BottConv(token_dim, sam_dims[i], mid_channels=token_dim // 4, kernel_size=3, padding=1, stride=1),
                nn.Upsample(size=self.savss_target_sizes[i], mode='bilinear', align_corners=False)
            )
            self.savss_decoder_heads.append(decoder_head)

    def forward_sam(self, x):
        """SAM/Hiera 分支的完整前向传播逻辑。"""
        sam_x_token, _ = self.sam_patch_embed(x)
        sam_x_token = sam_x_token + self.sam_pos_embed
        B, L_sam, C_sam = sam_x_token.shape
        H_sam = W_sam = int(L_sam**0.5)

        sam_features_raw = self.sam_encoder(sam_x_token, H_sam, W_sam)
        
        feats_sam = []
        for i in range(len(sam_features_raw)):
            sam_feature_detached = sam_features_raw[i].clone().detach()
            refined = self.refiners[i](sam_feature_detached)
            adapted = self.adapters[i](sam_feature_detached)
            feats_sam.append(refined + adapted)
        return tuple(feats_sam)

    def forward_mamba(self, x):
        """SAVSS/Mamba 分支的完整前向传播逻辑。"""
        mamba_x_token, _ = self.savss_patch_embed(x)
        mamba_x_token = mamba_x_token + self.savss_pos_embed
        B, L_mamba, C_mamba = mamba_x_token.shape
        # SAVSS分支的 patch_size 是 8, 所以 H 和 W 是 448/8 = 56
        H_mamba = W_mamba = int(L_mamba**0.5)

        feats_mamba = []
        for i, stage in enumerate(self.mamba_encoder):
            # SAVSS_Block 的输出与输入形状相同 (B, L, C)
            mamba_x_token = stage(mamba_x_token, (H_mamba, W_mamba))
            
            # 将 token 序列转换为 2D 特征图以送入解码头
            mamba_x_2d = mamba_x_token.transpose(1, 2).reshape(B, C_mamba, H_mamba, W_mamba)
            decoded_feat = self.savss_decoder_heads[i](mamba_x_2d)
            feats_mamba.append(decoded_feat)
        return tuple(feats_mamba)

    def forward(self, x_sam_input, x_mamba_input):
        """
        主干网络的前向传播，接收两个独立的输入，
        并返回两个维度完全匹配的并行特征金字塔。
        """
        feats_sam = self.forward_sam(x_sam_input)
        feats_mamba = self.forward_mamba(x_mamba_input)
        return (feats_sam, feats_mamba)
    
    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print(f"正在从 {pretrained} 加载预训练权重...")
        else:
            print("未提供预训练权重，将随机初始化。")
