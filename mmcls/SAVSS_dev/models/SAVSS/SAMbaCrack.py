# Author: Roy
# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# **重构**: 根据用户“方案一”要求，使用 U-Net 风格解码器替换 MFS，以强化分割细节。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

from mmcls.models.builder import BACKBONES
from mmcls.models.utils.embed import PatchEmbed

# --- 从项目根目录进行正确的绝对导入 ---
from mmcls.SAVSS_dev.models.samba_unet_modules.hiera import Hiera
from mmcls.SAVSS_dev.models.samba_unet_modules.refiner_adapter import DynamicFeatureFusionRefiner, MLPAdapter
from mmcls.SAVSS_dev.models.samba_unet_modules.hoacm import HOACM
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer
from models.DySample import DySample

# --- 【新增】U-Net 风格解码器模块，用于精细化特征重建 ---
class DecoderBlock(nn.Module):
    """U-Net 解码器的标准构建块。

    它包含一个上采样层，然后与来自编码器路径的跳跃连接特征进行拼接，
    最后通过一个卷积块进行处理。
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 上采样层，将深层特征图的尺寸放大两倍
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 卷积块，输入通道数 = 上采样后的通道数 + 跳跃连接的通道数
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        """
        Args:
            x (torch.Tensor): 来自更深（下一层）解码器块的特征图。
            skip (torch.Tensor): 来自编码器/融合器路径的、对应尺度的跳跃连接特征图。
        """
        x = self.upsample(x)
        # 拼接上采样后的特征和跳跃连接特征
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    """U-Net 风格的解码器，使用 DySample 进行高质量上采样。"""
    def __init__(self, decoder_channels):
        super().__init__()
        # decoder_channels 对应 c1, c2, c3, c4 的维度: [64, 128, 256, 512]
        c1_dim, c2_dim, c3_dim, c4_dim = decoder_channels

        # 解码器从最深层 (c4) 开始，逐级向上融合
        self.block1 = DecoderBlock(in_channels=c4_dim, skip_channels=c3_dim, out_channels=c3_dim)
        self.block2 = DecoderBlock(in_channels=c3_dim, skip_channels=c2_dim, out_channels=c2_dim)
        self.block3 = DecoderBlock(in_channels=c2_dim, skip_channels=c1_dim, out_channels=c1_dim)

        # 【新】使用两个 DySample 模块实现 4x 上采样 (112 -> 224 -> 448)
        self.dysample_x2 = DySample(c1_dim, scale=2, style='lp', groups=c1_dim)
        self.dysample_x4 = DySample(c1_dim, scale=2, style='lp', groups=c1_dim)

        # 【新】最终的通道融合卷积层
        self.final_conv = nn.Conv2d(c1_dim, 1, kernel_size=1)


    def forward(self, features):
        """
        Args:
            features (tuple): 包含4个尺度特征图的元组 (c4, c3, c2, c1)。
        """
        c4, c3, c2, c1 = features

        # 解码路径
        x = self.block1(c4, c3)    # 输出: (B, 256, 28, 28)
        x = self.block2(x, c2)    # 输出: (B, 128, 56, 56)
        x = self.block3(x, c1)    # 输出: (B, 64, 112, 112)

        # 【新】使用 DySample 进行高质量上采样
        x = self.dysample_x2(x)   # 输出: (B, 64, 224, 224)
        x = self.dysample_x4(x)   # 输出: (B, 64, 448, 448)

        # 【新】应用最终的卷积层生成 logits
        logits = self.final_conv(x) # 输出: (B, 1, 448, 448)

        return logits

logger = logging.getLogger(__name__)

@BACKBONES.register_module()
class SAMbaCrack(nn.Module):
    """最终的 SAMbaCrack 模型。"""

    def __init__(self, args, **kwargs):
        super().__init__()

        # --- 模型超参数定义 ---
        sam_dims = [96, 192, 384, 768]
        savss_dims = [64, 128, 256, 512]
        
        hiera_depths = getattr(args, 'hiera_depths', (2, 2, 6, 2))
        hiera_num_heads = getattr(args, 'hiera_num_heads', (3, 6, 12, 24))
        savss_drop_path_rate = getattr(args, 'savss_drop_path_rate', 0.1)
        savss_use_rms_norm = getattr(args, 'savss_use_rms_norm', True)
        savss_with_dwconv = getattr(args, 'savss_with_dwconv', True)

        # --- 1. Hiera (SAM) 分支 --- #
        self.sam_encoder = Hiera(
            img_size=args.load_height,
            patch_size=16,
            embed_dim=sam_dims[0],
            depths=hiera_depths,
            num_heads=hiera_num_heads,
            out_indices=(0, 1, 2, 3)
        )
        self.refiners = nn.ModuleList([DynamicFeatureFusionRefiner(dim=d) for d in sam_dims])
        self.adapters = nn.ModuleList([MLPAdapter(dim=d) for d in sam_dims])

        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # --- 2. SAVSS (Mamba) 分支 --- #
        self.savss_patch_embed = PatchEmbed(
            img_size=args.load_height,
            in_channels=3, 
            embed_dims=savss_dims[0], 
            conv_cfg={"kernel_size": 4, "stride": 4}
        )
        num_patches = (args.load_height // 4) ** 2
        self.savss_pos_embed = nn.Parameter(torch.zeros(1, num_patches, savss_dims[0]))
        nn.init.trunc_normal_(self.savss_pos_embed, std=0.02)

        self.mamba_encoder = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, savss_drop_path_rate, sum(hiera_depths))]

        for i in range(4):
            mamba_cfg = {'d_state': 16, 'expand': 2, 'dt_rank': 'auto', 'conv_size': 7}
            current_dpr = dpr[sum(hiera_depths[:i]):sum(hiera_depths[:i+1])][0]
            savss_block = SAVSS_Layer(
                embed_dims=savss_dims[i],
                use_rms_norm=savss_use_rms_norm,
                with_dwconv=savss_with_dwconv,
                drop_path_rate=current_dpr,
                mamba_cfg=mamba_cfg
            )
            downsample = nn.Sequential(
                nn.BatchNorm2d(savss_dims[i]),
                nn.Conv2d(savss_dims[i], savss_dims[i+1], kernel_size=2, stride=2)
            ) if i < 3 else nn.Identity()
            self.mamba_encoder.append(nn.ModuleDict({'block': savss_block, 'downsample': downsample}))

        # --- 3. 融合与解码器模块 --- #
        self.sam_adapters = nn.ModuleList([
            nn.Conv2d(sam_dims[i], savss_dims[i], kernel_size=1) for i in range(4)
        ])
        self.hoacms = nn.ModuleList([HOACM(dim=d) for d in savss_dims])
        
        # **【重构】**: 使用新的 UNetDecoder 替换掉旧的 MFS 解码器。
        # 这个解码器接收 HOACM 输出的特征维度列表，以构建经典的 U-Net 上采样路径。
        self.decoder = UNetDecoder(decoder_channels=savss_dims)

    def forward(self, x, stage=2):
        # --- SAVSS (Mamba) 分支总会运行 ---
        mamba_features = []
        mamba_x_token = self.savss_patch_embed(x)
        mamba_x_token = mamba_x_token + self.savss_pos_embed
        B, L, C = mamba_x_token.shape
        H = W = int(L**0.5)

        for i, mamba_stage in enumerate(self.mamba_encoder):
            mamba_x_token = mamba_stage['block'](mamba_x_token, (H, W))
            mamba_x_2d = mamba_x_token.transpose(1, 2).reshape(B, C, H, W)
            mamba_features.append(mamba_x_2d)
            
            if i < len(self.mamba_encoder) - 1:
                mamba_x_2d_down = mamba_stage['downsample'](mamba_x_2d)
                B, C, H, W = mamba_x_2d_down.shape
                mamba_x_token = mamba_x_2d_down.reshape(B, C, -1).transpose(1, 2)

        # --- 根据训练阶段选择数据流 ---
        if stage == 1:
            # 阶段一：直接将 Mamba 特征送入解码器
            decoder_input = (mamba_features[3], mamba_features[2], mamba_features[1], mamba_features[0])
        else: # 默认 stage 2
            # 阶段二：运行 Hiera 分支并进行融合
            with torch.no_grad():
                sam_features = self.sam_encoder(x)
            
            refined_sam_features = []
            for i in range(len(sam_features)):
                sam_feature_detached = sam_features[i].clone().detach()
                refined = self.refiners[i](sam_feature_detached)
                adapted = self.adapters[i](sam_feature_detached)
                refined_sam_features.append(refined + adapted)

            fused_features = []
            for i in range(4):
                sam_feat = refined_sam_features[i]
                mamba_feat = mamba_features[i]

                if sam_feat.shape[-2:] != mamba_feat.shape[-2:]:
                    sam_feat = F.interpolate(
                        sam_feat, 
                        size=mamba_feat.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                sam_feat_adapted = self.sam_adapters[i](sam_feat)
                fused = self.hoacms[i](sam_feat_adapted, mamba_feat)
                fused_features.append(fused)
            
            decoder_input = (fused_features[3], fused_features[2], fused_features[1], fused_features[0])

        # --- 解码器接收处理后的特征 ---
        logits = self.decoder(decoder_input)
        return logits

    def init_weights(self, pretrained=None):
        """初始化权重，特别是加载 Hiera 编码器的预训练权重。"""
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

                num_missing = len(missing_keys)
                num_unexpected = len(unexpected_keys)
                num_sam_encoder_params = len(self.sam_encoder.state_dict().keys())
                num_loaded_into_sam_encoder = num_sam_encoder_params - num_missing

                logger.info("--- SAM2 预训练权重加载摘要 ---")
                logger.info(f"  总计 {num_sam_encoder_params} 个 Hiera 编码器参数。")
                logger.info(f"  成功加载 {num_loaded_into_sam_encoder} 个参数。")
                
                if num_missing > 0:
                    logger.warning(f"  缺少 {num_missing} 个参数。")
                if num_unexpected > 0:
                    logger.warning(f"  有 {num_unexpected} 个意外的参数。")
                
                logger.info("--- 摘要结束 ---")
                logger.info("Hiera 编码器参数已被冻结，不会参与训练。")
            except Exception as e:
                logger.error(f"加载预训练权重时出错: {e}")
        else:
            logger.warning(f"在以下位置未找到预训练权重文件: {pretrained}。Hiera 将从头开始初始化。")
