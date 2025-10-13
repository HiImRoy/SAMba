# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class HieraBlock(nn.Module):
    """
    Hiera Transformer Block.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # [FIXED] 设置 batch_first=True，以正确处理 (B, L, C) 形状的输入张量。
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # 由于 self.attn 设置了 batch_first=True，现在可以直接处理 norm1(x) 的输出
        normed_x = self.norm1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    将 2x2 的相邻 token 合并为一个，空间分辨率减半，通道数加倍。
    """
    def __init__(self, in_channels, out_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        # [FIXED] LayerNorm 应该在 4*in_channels 上操作，因为它在 token 合并之后、降维之前被调用。
        self.norm = norm_layer(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)

        # 将 2x2 的相邻 token 在通道维度上拼接
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 在降维之前进行归一化
        x = self.norm(x)
        # 线性降维
        x = self.reduction(x)
        return x


class HieraStage(nn.Module):
    """
    一个 Hiera 阶段，包含多个 HieraBlock 和一个可选的下采样层。
    """
    def __init__(self, in_channels, out_channels, depth, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=True):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            HieraBlock(dim=in_channels, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer)
            for i in range(depth)])
        if downsample:
            self.downsample = PatchMerging(in_channels, out_channels, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class Hiera(nn.Module):
    """
    [MODIFIED] Hiera 模型，现在接收一个 token 序列作为输入。
    """
    def __init__(self, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, out_indices=(0, 1, 2, 3)):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        in_channels = embed_dim
        for i in range(self.num_layers):
            out_channels = embed_dim * 2 ** (i + 1)
            stage = HieraStage(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_layers - 1))
            self.stages.append(stage)
            in_channels = out_channels

        # 为每个输出阶段添加归一化层
        for i in out_indices:
            layer = norm_layer(embed_dim * 2 ** i)
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def forward(self, x_token, H, W):
        """
        [MODIFIED] Hiera 的前向传播，直接处理 token 序列。
        """
        x = x_token
        outs = []
        for i, stage in enumerate(self.stages):
            x, H, W, x_down, Wh, Ww = stage(x, H, W)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                out = x_out.view(-1, H, W, self.embed_dim * 2 ** i).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
            x = x_down
            H, W = Wh, Ww

        return tuple(outs)
