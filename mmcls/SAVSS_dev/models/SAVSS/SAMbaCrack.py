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

# --- 从项目根目录进行正确的绝对导入 ---
from mmcls.SAVSS_dev.models.samba_unet_modules.hiera import Hiera
from mmcls.SAVSS_dev.models.samba_unet_modules.refiner_adapter import DynamicFeatureFusionRefiner, MLPAdapter
from mmcls.SAVSS_dev.models.samba_unet_modules.hoacm import HOACM
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer

# --- 从 models/GBC.py 内联 GBC 和 BottConv (MFS 和 SAVSS_layer 使用) ---
class BottConv(nn.Module):
    """瓶颈卷积块。"""
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, padding, stride):
        super(BottConv, self).__init__()
        self.bott_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bott_conv(x)

class GBC(nn.Module):
    """全局瓶颈卷积块。"""
    def __init__(self, in_channels, out_channels=None, intermediate_channels=None, norm_type=None):
        super(GBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.intermediate_channels = intermediate_channels if intermediate_channels is not None else in_channels // 4

        self.gwc = BottConv(self.in_channels, self.out_channels, self.intermediate_channels, 3, 1, 1)
        if norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(self.out_channels)
        else:
            self.norm = nn.BatchNorm2d(self.out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.gwc(x)
        x = self.act(self.norm(x))
        return x + res

# --- 从 models/DySample.py 内联 DySample (已修复) ---
def normal_init(module, mean=0, std=1, bias=0):
    """正态分布初始化。"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    """常量初始化。"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    """动态采样模块，用于上采样。"""
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        # 使用 reshape 替代 view
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        # 在 transpose 后添加 .contiguous() 以确保内存连续
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')
                             ).transpose(1, 2).contiguous().unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        # 使用 reshape 替代 view
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        # 使用 reshape 替代 view
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        # 使用 reshape 替代 view
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

# --- 从 models/MFS.py 内联 MFS ---
class MLP(nn.Module):
    """多层感知机，用于特征维度映射。"""
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class MFS(nn.Module):
    """多尺度特征分割头 (MFSHead)。"""
    def __init__(self, embedding_dim):
        super(MFS, self).__init__()

        self.embedding_dim = embedding_dim
        # 将不同层级的特征映射到相同的 embedding_dim
        self.linear_c4 = MLP(input_dim=256, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=32, embed_dim=embedding_dim)
        self.GBC_C = GBC(embedding_dim*4)
        self.linear_fuse = BottConv(embedding_dim*4, embedding_dim, embedding_dim//8, kernel_size=1, padding=0, stride=1)

        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1, padding=0, stride=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

        # 动态上采样模块，用于将不同尺度的特征图统一到最大尺寸
        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)

    def forward(self, inputs, final_size):
        c4, c3, c2, c1 = inputs # c4是最高层(最小)，c1是最低层(最大)
        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(c4.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c4 = self.DySample_C_8(out_c4) # 上采样8倍

        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(c3.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c3 = self.DySample_C_4(out_c3) # 上采样4倍

        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(c2.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
        out_c2 = self.DySample_C_2(out_c2) # 上采样2倍

        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(c1.reshape(b, c, h*w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)

        # 拼接所有在 112x112 尺度上的特征图
        fused_low_res = torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1)

        # --- MODIFIED: Upsample to full resolution BEFORE final fusion ---
        full_res_features = F.interpolate(fused_low_res, size=final_size, mode='bilinear', align_corners=False)

        # --- MODIFIED: Perform all final computations on the full resolution feature map ---
        out_c = self.GBC_C(full_res_features)
        out_c = self.linear_fuse(out_c)

        out_c = self.dropout(out_c)
        x = self.linear_pred_1(self.linear_pred(out_c))

        # No final interpolation needed as we are already at full resolution
        return x

logger = logging.getLogger(__name__)

@BACKBONES.register_module()
class SAMbaCrack(nn.Module):
    """最终的 SAMbaCrack 模型。"""

    def __init__(self, args, **kwargs):
        super().__init__()

        # --- 模型超参数定义 ---
        hiera_embed_dim = getattr(args, 'hiera_embed_dim', 96)
        hiera_depths = getattr(args, 'hiera_depths', (2, 2, 6, 2))
        hiera_num_heads = getattr(args, 'hiera_num_heads', (3, 6, 12, 24))
        savss_dims = getattr(args, 'savss_dims', (96, 192, 384, 768))
        mfs_embed_dim = getattr(args, 'mfs_embed_dim', 16)
        savss_drop_path_rate = getattr(args, 'savss_drop_path_rate', 0.1)
        savss_use_rms_norm = getattr(args, 'savss_use_rms_norm', True)
        savss_with_dwconv = getattr(args, 'savss_with_dwconv', True)

        # --- 1. Hiera (SAM) 分支 --- #
        self.sam_encoder = Hiera(
            img_size=args.load_height,
            patch_size=16, # patch=16 会产生较小的特征图
            embed_dim=hiera_embed_dim,
            depths=hiera_depths,
            num_heads=hiera_num_heads,
            out_indices=(0, 1, 2, 3)
        )
        sam_dims = [hiera_embed_dim * (2**i) for i in range(4)]
        # SAM 特征的精炼器和适配器
        self.refiners = nn.ModuleList([DynamicFeatureFusionRefiner(dim=d) for d in sam_dims])
        self.adapters = nn.ModuleList([MLPAdapter(dim=d) for d in sam_dims])

        # --- 为 PEFT 冻结 SAM 编码器 --- #
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # --- 2. SAVSS (Mamba) 分支 --- #
        self.savss_patch_embed = PatchEmbed(
            img_size=args.load_height,
            in_channels=3, 
            embed_dims=savss_dims[0], 
            conv_cfg={"kernel_size": 4, "stride": 4} # patch=4 会产生较大的特征图
        )
        num_patches = (args.load_height // 4) ** 2
        self.savss_pos_embed = nn.Parameter(torch.zeros(1, num_patches, savss_dims[0]))
        nn.init.trunc_normal_(self.savss_pos_embed, std=0.02)

        self.mamba_encoder = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, savss_drop_path_rate, sum(hiera_depths))]

        # 构建4个阶段的 Mamba 编码器
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
            # 每个阶段后的下采样层
            downsample = nn.Sequential(
                nn.BatchNorm2d(savss_dims[i]),
                nn.Conv2d(savss_dims[i], savss_dims[i+1], kernel_size=2, stride=2)
            ) if i < 3 else nn.Identity()
            self.mamba_encoder.append(nn.ModuleDict({'block': savss_block, 'downsample': downsample}))

        # --- 3. 融合与解码器模块 --- #
        # HOACM 要求输入具有相同的通道维度，这里 sam_dims 和 savss_dims 是一致的
        self.hoacms = nn.ModuleList([HOACM(dim=d) for d in sam_dims])
        # 投影层，用于在送入解码器前调整通道数
        mfs_dims = [32, 64, 128, 256]
        self.projections = nn.ModuleList([nn.Conv2d(sam_dims[i], mfs_dims[i], kernel_size=1) for i in range(4)])
        self.decoder = MFS(embedding_dim=mfs_embed_dim)

    def forward(self, x):
        # --- Hiera 分支前向传播 ---
        # 注意：因为编码器被冻结，这里应该在评估模式下运行以禁用 Dropout 等
        with torch.no_grad():
            sam_features = self.sam_encoder(x) # 包含4个阶段特征图的元组
        
        refined_sam_features = []
        for i in range(len(sam_features)):
            # .clone().detach() 是为了确保反向传播的计算图不会流回冻结的编码器
            sam_feature_detached = sam_features[i].clone().detach()
            refined = self.refiners[i](sam_feature_detached)
            adapted = self.adapters[i](sam_feature_detached)
            refined_sam_features.append(refined + adapted)

        # --- SAVSS 分支前向传播 ---
        mamba_features = []
        mamba_x_token = self.savss_patch_embed(x)
        mamba_x_token = mamba_x_token + self.savss_pos_embed
        B, L, C = mamba_x_token.shape
        H = W = int(L**0.5)

        for i, stage in enumerate(self.mamba_encoder):
            mamba_x_token = stage['block'](mamba_x_token, (H, W))
            # 将 token 重塑为 2D 特征图，用于融合和下采样
            mamba_x_2d = mamba_x_token.transpose(1, 2).reshape(B, C, H, W)
            mamba_features.append(mamba_x_2d)
            
            if i < len(self.mamba_encoder) - 1:
                mamba_x_2d_down = stage['downsample'](mamba_x_2d)
                B, C, H, W = mamba_x_2d_down.shape
                mamba_x_token = mamba_x_2d_down.reshape(B, C, -1).transpose(1, 2)

        # --- 特征融合、投影和解码 ---
        projected_features = []
        for i in range(4):
            sam_feat = refined_sam_features[i]
            mamba_feat = mamba_features[i]

            # --- 维度对齐 --- #
            # 将较小的特征图 (来自 Hiera) 上采样以匹配较大的特征图 (来自 Mamba)。
            if sam_feat.shape[-2:] != mamba_feat.shape[-2:]:
                sam_feat = F.interpolate(
                    sam_feat, 
                    size=mamba_feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )

            fused = self.hoacms[i](sam_feat, mamba_feat) # HOACM 融合
            projected = self.projections[i](fused) # 投影到解码器所需维度
            projected_features.append(projected)

        # 将特征按 C4, C3, C2, C1 的顺序送入解码器
        decoder_input = (projected_features[3], projected_features[2], projected_features[1], projected_features[0])
        
        # --- MODIFIED: Pass final size to decoder and get full-resolution output directly ---
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

                # Load state_dict with strict=False to get missing and unexpected keys
                missing_keys, unexpected_keys = self.sam_encoder.load_state_dict(state_dict, strict=False)

                num_missing = len(missing_keys)
                num_unexpected = len(unexpected_keys)
                
                # Calculate the number of keys that were actually loaded into sam_encoder
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
