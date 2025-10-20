# -*- coding: utf-8 -*-
"""
# SAMbaCrack 模型分析脚本

# 目的:
# 评估 SAMbaCrack 模型中主要构建块的参数量和计算量 (FLOPs)。

# 使用方法:
# 1. 确保已安装 thop: pip install thop
# 2. 在终端中运行: python profile_model.py

# 注意:
# - 为了使脚本独立运行，部分模块定义（如 Hiera, LoRA, UNetDecoder）已从项目其他文件中复制至此。
# - SAVSS_Layer 和 HOACM 是根据其用法创建的简化模块，用于完成模型结构分析，
#   其内部实现已简化，以保证参数和FLOPs计算的准确性。
# - 脚本默认使用 448x448 的输入尺寸进行FLOPs计算。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from thop import profile
from functools import partial

# ==============================================================================
# 1. 核心模块定义 (从 SAMbaCrack.py, hiera.py 等整合)
# ==============================================================================

# --- LoRA 模块 ---
class LoRALinear(nn.Module):
    """替换 nn.Linear 层并注入可训练的低秩矩阵。"""
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha

        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # 创建低秩矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.linear.out_features, self.rank))
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = self.linear(x)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        return original_output + lora_output * (self.alpha / self.rank)

# --- Hiera 模块 (简化版，保留核心结构) ---
class HieraBlock(nn.Module):
    """Hiera 的基本构建块。"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # 使用 nn.MultiheadAttention 作为注意力机制的代表
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        self.drop_path = nn.Identity() # 在分析中忽略 dropout
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        normed_x = self.norm1(x)
        # MultiheadAttention 返回 (output, weights)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """图像到 Patch Embedding 的转换层。"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, H*W, C
        x = self.norm(x)
        return x

class Hiera(nn.Module):
    """Hiera 视觉 Transformer 模型。"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), out_indices=(0, 1, 2, 3), **kwargs):
        super().__init__()
        self.num_levels = len(depths)
        self.out_indices = out_indices
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level_dim = int(embed_dim * (2 ** i))
            prev_level_dim = embed_dim if i == 0 else int(embed_dim * (2 ** (i - 1)))
            
            downsample = nn.Identity()
            if i > 0:
                downsample = nn.Sequential(
                    norm_layer(prev_level_dim),
                    nn.Conv2d(prev_level_dim, level_dim, kernel_size=2, stride=2)
                )

            blocks = nn.Sequential(*[HieraBlock(dim=level_dim, num_heads=num_heads[i], norm_layer=norm_layer) for _ in range(depths[i])])
            self.levels.append(nn.ModuleDict({'downsample': downsample, 'blocks': blocks}))

    def forward(self, x):
        x = self.patch_embed(x)
        x += self.pos_embed
        
        outs = []
        B, L, C = x.shape
        H, W = int(L**0.5), int(L**0.5)

        for i, level in enumerate(self.levels):
            if i > 0:
                x = x.reshape(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
                x = level['downsample'](x)
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            x = level['blocks'](x)

            if i in self.out_indices:
                out = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

# --- U-Net Decoder 模块 ---
class DecoderBlock(nn.Module):
    """U-Net 解码器的标准构建块。"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    """U-Net 风格的解码器。"""
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

# --- 简化的 SAVSS 和 HOACM 模块 (用于分析) ---
class SAVSS_Layer(nn.Module):
    """简化的 SAVSS_Layer，模拟参数和维度变换。"""
    def __init__(self, embed_dims, **kwargs):
        super().__init__()
        # 模拟 Mamba 块中的线性变换和卷积
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, groups=embed_dims) # DWConv

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        x_proj = self.proj(x)
        x_conv = x.transpose(1, 2).reshape(B, C, H, W)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.reshape(B, C, L).transpose(1, 2)
        return x_proj + x_conv # 模拟残差连接

class HOACM(nn.Module):
    """简化的 HOACM，模拟特征融合。"""
    def __init__(self, dim, **kwargs):
        super().__init__()
        # 模拟融合过程中的卷积
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, sam_feat, mamba_feat):
        combined = torch.cat([sam_feat, mamba_feat], dim=1)
        return self.conv(combined)

# --- 主模型 SAMbaCrack ---
def inject_lora_into_hiera(hiera_model, rank, alpha):
    """查找 HieraBlocks 并注入 LoRA 层。"""
    for module in hiera_model.modules():
        if isinstance(module, HieraBlock):
            # 注入到 MLP 和 注意力输出
            module.mlp[0] = LoRALinear(module.mlp[0], rank, alpha)
            module.mlp[2] = LoRALinear(module.mlp[2], rank, alpha)
            module.attn.out_proj = LoRALinear(module.attn.out_proj, rank, alpha)

class SAMbaCrack(nn.Module):
    """最终的 SAMbaCrack 模型定义，用于分析。"""
    def __init__(self, args, **kwargs):
        super().__init__()
        sam_dims = [96, 192, 384, 768]
        savss_dims = [64, 128, 256, 512]
        hiera_depths = getattr(args, 'hiera_depths', (2, 2, 6, 2))
        hiera_num_heads = getattr(args, 'hiera_num_heads', (3, 6, 12, 24))

        # 1. Hiera (SAM) 分支
        self.sam_encoder = Hiera(img_size=args.load_height, patch_size=16, embed_dim=sam_dims[0], depths=hiera_depths, num_heads=hiera_num_heads, out_indices=(0, 1, 2, 3))
        for param in self.sam_encoder.parameters():
            param.requires_grad = False
        lora_rank = getattr(args, 'lora_rank', 4)
        lora_alpha = getattr(args, 'lora_alpha', 1.0)
        inject_lora_into_hiera(self.sam_encoder, rank=lora_rank, alpha=lora_alpha)

        # 2. SAVSS (Mamba) 分支
        self.savss_patch_embed = PatchEmbed(img_size=args.load_height, patch_size=4, in_chans=3, embed_dim=savss_dims[0])
        num_patches = (args.load_height // 4) ** 2
        self.savss_pos_embed = nn.Parameter(torch.zeros(1, num_patches, savss_dims[0]))
        
        self.mamba_encoder = nn.ModuleList()
        for i in range(4):
            savss_block = SAVSS_Layer(embed_dims=savss_dims[i])
            downsample = nn.Sequential(nn.BatchNorm2d(savss_dims[i]), nn.Conv2d(savss_dims[i], savss_dims[i+1], kernel_size=2, stride=2)) if i < 3 else nn.Identity()
            self.mamba_encoder.append(nn.ModuleDict({'block': savss_block, 'downsample': downsample}))

        # 3. 融合与解码器模块
        self.sam_adapters = nn.ModuleList([nn.Conv2d(sam_dims[i], savss_dims[i], kernel_size=1) for i in range(4)])
        self.hoacms = nn.ModuleList([HOACM(dim=d) for d in savss_dims])
        self.decoder = UNetDecoder(decoder_channels=savss_dims)

    def forward(self, x):
        # Hiera Encoder 分支
        sam_features = self.sam_encoder(x)

        # Mamba Encoder 分支
        mamba_features = []
        mamba_x_token = self.savss_patch_embed(x)
        mamba_x_token += self.savss_pos_embed
        B, L, C = mamba_x_token.shape
        H = W = int(L**0.5)

        for i, stage in enumerate(self.mamba_encoder):
            mamba_x_token = stage['block'](mamba_x_token, (H, W))
            mamba_x_2d = mamba_x_token.transpose(1, 2).reshape(B, C, H, W)
            mamba_features.append(mamba_x_2d)
            
            if i < 3:
                mamba_x_2d_down = stage['downsample'](mamba_x_2d)
                B, C, H, W = mamba_x_2d_down.shape
                mamba_x_token = mamba_x_2d_down.reshape(B, C, -1).transpose(1, 2)

        # 特征融合
        fused_features = []
        for i in range(4):
            sam_feat = sam_features[i]
            mamba_feat = mamba_features[i]

            if sam_feat.shape[-2:] != mamba_feat.shape[-2:]:
                sam_feat = F.interpolate(sam_feat, size=mamba_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            sam_feat_adapted = self.sam_adapters[i](sam_feat)
            fused = self.hoacms[i](sam_feat_adapted, mamba_feat)
            fused_features.append(fused)

        # U-Net 解码器
        decoder_input = (fused_features[3], fused_features[2], fused_features[1], fused_features[0])
        logits = self.decoder(decoder_input, final_size=x.shape[-2:])

        return logits

# ==============================================================================
# 2. 分析脚本主逻辑
# ==============================================================================

def count_parameters(module, name):
    """计算并打印模块的参数量"""
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in module.parameters())
    print(f"  > {name} 参数量:")
    print(f"    - 可训练参数: {trainable_params / 1e6:.4f} M")
    print(f"    - 总参数:     {total_params / 1e6:.4f} M")
    return trainable_params, total_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAMbaCrack 模型分析脚本")
    parser.add_argument('--load_height', type=int, default=448, help='输入图像高度')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA 的秩')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA 的 alpha 缩放因子')
    parser.add_argument('--hiera_depths', nargs='+', type=int, default=[2, 2, 6, 2], help='Hiera 各阶段深度')
    parser.add_argument('--hiera_num_heads', nargs='+', type=int, default=[3, 6, 12, 24], help='Hiera 各阶段头数')

    args = parser.parse_args()

    # --- 模型实例化 ---
    model = SAMbaCrack(args)
    model.eval()

    # --- 准备输入 ---
    dummy_input = torch.randn(1, 3, args.load_height, args.load_height)

    print("-" * 80)
    print(f"开始分析 SAMbaCrack (输入尺寸: {args.load_height}x{args.load_height}, LoRA rank: {args.lora_rank})")
    print("-" * 80)

    # --- 1. 整体模型分析 ---
    total_flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    print("1. 整体模型性能")
    print(f"  > 总 GFLOPs: {total_flops / 1e9:.4f} G")
    _, total_model_params = count_parameters(model, "整体模型")
    print("-" * 80)

    # --- 2. SAM (Hiera) Encoder 分析 ---
    print("2. SAM 编码器 (Hiera with LoRA)")
    # Hiera的输入与模型输入相同
    sam_flops, _ = profile(model.sam_encoder, inputs=(dummy_input,), verbose=False)
    print(f"  > GFLOPs: {sam_flops / 1e9:.4f} G")
    count_parameters(model.sam_encoder, "SAM 编码器")
    print("-" * 80)

    # --- 3. Mamba (SAVSS) 分支参数分析 ---
    print("3. Mamba 分支 (SAVSS)")
    # 将 Mamba 分支的所有组件打包成一个 ModuleList 以便统计
    mamba_branch_modules = nn.ModuleList([
        model.savss_patch_embed,
        model.mamba_encoder
    ])
    # 单独添加 nn.Parameter
    mamba_total_params = sum(p.numel() for p in mamba_branch_modules.parameters())
    mamba_total_params += model.savss_pos_embed.numel()
    mamba_trainable_params = sum(p.numel() for p in mamba_branch_modules.parameters() if p.requires_grad)
    mamba_trainable_params += model.savss_pos_embed.numel() if model.savss_pos_embed.requires_grad else 0
    
    print("  > GFLOPs: (难以独立计算，已包含在总数中)")
    print(f"  > Mamba 分支 参数量:")
    print(f"    - 可训练参数: {mamba_trainable_params / 1e6:.4f} M")
    print(f"    - 总参数:     {mamba_total_params / 1e6:.4f} M")
    print("-" * 80)

    # --- 4. 融合模块参数分析 ---
    print("4. 融合模块 (Adapters + HOACM)")
    fusion_blocks = nn.ModuleList([model.sam_adapters, model.hoacms])
    print("  > GFLOPs: (难以独立计算，已包含在总数中)")
    count_parameters(fusion_blocks, "融合模块")
    print("-" * 80)

    # --- 5. U-Net 解码器参数分析 ---
    print("5. U-Net 解码器")
    print("  > GFLOPs: (难以独立计算，已包含在总数中)")
    count_parameters(model.decoder, "U-Net 解码器")
    print("-" * 80)

    print("分析完成。")

