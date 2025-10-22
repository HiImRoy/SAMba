# Author: Roy
# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# **重构**: 根据用户“方案一”要求，使用 U-Net 风格解码器替换 MFS，以强化分割细节。
# **重构**: 使用 Sequential Adapter 替换 LoRA，实现更清晰的参数高效微调。
# **修正**: 修正了 Adapter 的 forward 签名以匹配 HieraBlock。
# **重构**: 使用 FCM 模块替换 HOACM 模块作为特征融合器。
# **修正**: 修复了 Hiera 模型中 Adapter 注入的错误路径。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import math

from mmcls.models.builder import BACKBONES
from mmcls.models.utils.embed import PatchEmbed

# --- 从项目根目录进行正确的绝对导入 ---
from mmcls.SAVSS_dev.models.samba_unet_modules.hiera import Hiera, HieraBlock
from models.fcm import FCM
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer


# --- 【重构】使用 Sequential Adapter 进行微调 ---
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        # 【修正】HieraBlock中的注意力是nn.MultiheadAttention，没有.qkv属性。
        # 通过其norm层的normalized_shape来安全地获取维度信息。
        dim = blk.norm1.normalized_shape[0]
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),  # Bottleneck dimension of 32
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Adapter 的前向传播逻辑。
        【修正】HieraBlock 的 forward 只需要 x 作为输入。
        """
        # 1. 基于输入 x 生成一个 prompt
        prompt = self.prompt_learn(x)
        # 2. 将 prompt 添加到原始输入上
        promped_x = x + prompt
        # 3. 将增强后的输入传递给原始的、被冻结的 block
        net = self.block(promped_x)
        return net


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
    """U-Net 风格的解码器，逐级融合多尺度特征以重建分割细节。"""
    def __init__(self, decoder_channels):
        super().__init__()
        # decoder_channels 对应 c1, c2, c3, c4 的维度: [64, 128, 256, 512]
        c1_dim, c2_dim, c3_dim, c4_dim = decoder_channels

        # 解码器从最深层 (c4) 开始，逐级向上融合
        # Block 1: 融合 c4 和 c3
        self.block1 = DecoderBlock(in_channels=c4_dim, skip_channels=c3_dim, out_channels=c3_dim)
        # Block 2: 融合 block1 的输出和 c2
        self.block2 = DecoderBlock(in_channels=c3_dim, skip_channels=c2_dim, out_channels=c2_dim)
        # Block 3: 融合 block2 的输出和 c1
        self.block3 = DecoderBlock(in_channels=c2_dim, skip_channels=c1_dim, out_channels=c1_dim)

        # 分割头，将最终的高分辨率特征图转换为单通道的 logits
        self.segmentation_head = nn.Conv2d(c1_dim, 1, kernel_size=1)

    def forward(self, features, final_size):
        """
        Args:
            features (tuple): 包含4个尺度特征图的元组 (c4, c3, c2, c1)。
            final_size (tuple): 最终输出 logits 需要被上采样到的目标尺寸 (H, W)。
        """
        c4, c3, c2, c1 = features

        # 解码路径
        x = self.block1(c4, c3)    # 输出尺寸与 c3 相同
        x = self.block2(x, c2)    # 输出尺寸与 c2 相同
        x = self.block3(x, c1)    # 输出尺寸与 c1 相同 (e.g., 112x112)

        # 应用分割头
        logits = self.segmentation_head(x)

        # 将 logits 上采样到原始输入图像的尺寸
        logits = F.interpolate(logits, size=final_size, mode='bilinear', align_corners=False)

        return logits

logger = logging.getLogger(__name__)

@BACKBONES.register_module()
class SAMbaCrack(nn.Module):
    """最终的 SAMbaCrack 模型。"""

    def __init__(self, args, **kwargs):
        super().__init__()

        # --- 模型超参数定义 ---
        # model #Channels #Blocks #Heads FLOPs Param
        # Hiera-T [96-192-384-768] [1-2-7-2] [1-2-4-8] 5G 28M
        # Hiera-S [96-192-384-768] [1-2-11-2] [1-2-4-8] 6G 35M
        # Hiera-B [96-192-384-768] [2-3-16-3] [1-2-4-8] 9G 52M

        # Hiera-B+ [112-224-448-896] [2-3-16-3] [2-4-8-16] 13G 70M

        # Hiera-L [144-288-576-1152] [2-6-36-4] [2-4-8-16] 40G 214M
        # Hiera-H [256-512-1024-2048] [2-6-36-4] [4-8-16-32] 125G 673M

        # 正儿八经但是效果不好的超参数
        self.sam_dims = [112, 224, 448, 896]
        self.savss_dims = [64, 128, 256, 512]
        self.hiera_depths = getattr(args, 'hiera_depths', (2, 3, 16, 3))

        # 莫名其妙但是就是效果好的超参数
        # self.sam_dims = [96, 192, 384, 768]
        # self.savss_dims = [64, 128, 256, 512]
        # self.hiera_depths = getattr(args, 'hiera_depths', (2, 2, 6, 2))

        self.hiera_num_heads = getattr(args, 'hiera_num_heads', (2, 4, 8, 16))
        savss_drop_path_rate = getattr(args, 'savss_drop_path_rate', 0.1)
        savss_use_rms_norm = getattr(args, 'savss_use_rms_norm', True)
        savss_with_dwconv = getattr(args, 'savss_with_dwconv', True)

        # Add patch sizes and fcm output dims as attributes
        self.hiera_patch_size = 16 # from Hiera init
        self.savss_patch_size = 4 # from PatchEmbed init (conv_cfg={"kernel_size": 4, "stride": 4})
        self.fcm_output_dims = self.savss_dims # FCM output dims are savss_dims

        # --- 1. Hiera (SAM) 分支 --- #
        self.sam_encoder = Hiera(
            img_size=args.load_height,
            patch_size=self.hiera_patch_size,
            embed_dim=self.sam_dims[0],
            depths=self.hiera_depths,
            num_heads=self.hiera_num_heads,
            out_indices=(0, 1, 2, 3)
        )
        
        # --- 【重构】应用 Sequential Adapter ---
        # 首先，冻结所有 SAM Encoder 的参数
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # 【修正】然后，根据正确的 Hiera 结构，将每个 HieraBlock 替换为 Adapter 包裹的块
        for level in self.sam_encoder.levels:
            # 每个 level 是一个 ModuleDict，其 'blocks' 键对应一个 nn.Sequential 容器
            sequential_blocks = level['blocks']
            for i in range(len(sequential_blocks)):
                # 将 Sequential 容器中的 HieraBlock 替换为 Adapter(HieraBlock)
                sequential_blocks[i] = Adapter(sequential_blocks[i])
        
        # 打印可训练参数数量以供验证
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"SAMbaCrack with Sequential Adapter enabled. Trainable parameters: {trainable_params/1e6:.2f}M")


        # --- 2. SAVSS (Mamba) 分支 --- #
        self.savss_patch_embed = PatchEmbed(
            img_size=args.load_height,
            in_channels=3, 
            embed_dims=self.savss_dims[0], 
            conv_cfg={"kernel_size": self.savss_patch_size, "stride": self.savss_patch_size}
        )
        num_patches = (args.load_height // self.savss_patch_size) ** 2
        self.savss_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.savss_dims[0]))
        nn.init.trunc_normal_(self.savss_pos_embed, std=0.02)

        self.mamba_encoder = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, savss_drop_path_rate, sum(self.hiera_depths))]

        for i in range(4):
            mamba_cfg = {'d_state': 16, 'expand': 2, 'dt_rank': 'auto', 'conv_size': 7}
            current_dpr = dpr[sum(self.hiera_depths[:i]):sum(self.hiera_depths[:i+1])][0]
            savss_block = SAVSS_Layer(
                embed_dims=self.savss_dims[i],
                use_rms_norm=savss_use_rms_norm,
                with_dwconv=savss_with_dwconv,
                drop_path_rate=current_dpr,
                mamba_cfg=mamba_cfg
            )
            downsample = nn.Sequential(
                nn.BatchNorm2d(self.savss_dims[i]),
                nn.Conv2d(self.savss_dims[i], self.savss_dims[i+1], kernel_size=2, stride=2)
            ) if i < 3 else nn.Identity()
            self.mamba_encoder.append(nn.ModuleDict({'block': savss_block, 'downsample': downsample}))

        # --- 3. 融合与解码器模块 --- #
        self.sam_adapters = nn.ModuleList([
            nn.Conv2d(self.sam_dims[i], self.savss_dims[i], kernel_size=1) for i in range(4)
        ])
        # 【重构】使用 FCM 替换 HOACM
        self.fcms = nn.ModuleList([FCM(dim=d) for d in self.savss_dims])
        
        # **【重构】**: 使用新的 UNetDecoder 替换掉旧的 MFS 解码器。
        self.decoder = UNetDecoder(decoder_channels=self.savss_dims)

    def forward(self, x):
        # Hiera Encoder 现在通过 Adapter 模块进行微调，前向传播调用保持不变
        sam_features = self.sam_encoder(x)

        mamba_features = []
        mamba_x_token = self.savss_patch_embed(x)
        mamba_x_token = mamba_x_token + self.savss_pos_embed
        B, L, C = mamba_x_token.shape
        H = W = int(L**0.5)

        for i, stage in enumerate(self.mamba_encoder):
            mamba_x_token = stage['block'](mamba_x_token, (H, W))
            mamba_x_2d = mamba_x_token.transpose(1, 2).reshape(B, C, H, W)
            mamba_features.append(mamba_x_2d)
            
            if i < len(self.mamba_encoder) - 1:
                mamba_x_2d_down = stage['downsample'](mamba_x_2d)
                B, C, H, W = mamba_x_2d_down.shape
                mamba_x_token = mamba_x_2d_down.reshape(B, C, -1).transpose(1, 2)

        fused_features = []
        for i in range(4):
            sam_feat = sam_features[i]
            mamba_feat = mamba_features[i]

            if sam_feat.shape[-2:] != mamba_feat.shape[-2:]:
                sam_feat = F.interpolate(
                    sam_feat, 
                    size=mamba_feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            sam_feat_adapted = self.sam_adapters[i](sam_feat)
            # 【重构】调用 FCM 模块
            fused = self.fcms[i](sam_feat_adapted, mamba_feat)
            fused_features.append(fused)

        # 将融合后的特征按 (c4, c3, c2, c1) 顺序送入新的 U-Net 解码器。
        decoder_input = (fused_features[3], fused_features[2], fused_features[1], fused_features[0])
        logits = self.decoder(decoder_input, final_size=x.shape[-2:])

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

                # 加载权重时，由于我们将 HieraBlock 替换为了 Adapter(HieraBlock)，
                # 预训练模型中不存在 Adapter 的参数 (prompt_learn)，所以我们设置 strict=False。
                missing_keys, unexpected_keys = self.sam_encoder.load_state_dict(state_dict, strict=False)

                # 过滤掉我们预期会缺失的 Adapter 参数，只报告非预期的缺失
                missing_keys_non_adapter = [k for k in missing_keys if 'prompt_learn' not in k]
                
                num_missing = len(missing_keys_non_adapter)
                num_unexpected = len(unexpected_keys)
                
                logger.info("--- SAM2 预训练权重加载摘要 (Adapter enabled) ---")
                if num_missing > 0:
                    logger.warning(f"  警告: 缺少 {num_missing} 个非 Adapter 参数。")
                if num_unexpected > 0:
                    logger.warning(f"  警告: 有 {num_unexpected} 个意外的参数。")
                if num_missing == 0 and num_unexpected == 0:
                    logger.info("  所有非 Adapter 参数均已成功加载。")
                
                logger.info("--- 摘要结束 ---")
                logger.info("Hiera 编码器原始参数已被冻结，只有 Adapter 参数会参与训练。")
            except Exception as e:
                logger.error(f"加载预训练权重时出错: {e}")
        else:
            logger.warning(f"在以下位置未找到预训练权重文件: {pretrained}。Hiera 将从头开始初始化。")
