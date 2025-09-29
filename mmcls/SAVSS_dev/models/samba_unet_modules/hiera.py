# Copyright (c) Facebook, Inc. and its affiliates.
# All rights-reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The code is modified from the official Hiera implementation
# to fit the mmcls framework and the project's needs.

import torch
import torch.nn as nn
from functools import partial

from mmcls.models.builder import BACKBONES
from mmcls.models.utils.embed import PatchEmbed
from mmcls.utils import get_root_logger


class HieraBlock(nn.Module):
    """
    A Hiera block that can be configured for either masked autoencoding or classification.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        # Multi-head self-attention
        normed_x = self.norm1(x)
        x = x + self.drop_path(self.attn(normed_x, normed_x, normed_x)[0])
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@BACKBONES.register_module()
class Hiera(nn.Module):
    """
    Hiera: A Hierarchical Vision Transformer.
    This implementation is adapted for mmcls and to be used as a SAM2 encoder backbone.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(4, 8, 16, 32),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=None,
        act_layer=None,
        out_indices=(0, 1, 2, 3),
        **kwargs,
    ):
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.out_indices = out_indices
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6) if norm_layer is None else norm_layer
        act_layer = nn.GELU if act_layer is None else act_layer

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_chans,
            embed_dims=embed_dim,
            conv_cfg={
                "kernel_size": patch_size,
                "stride": patch_size,
            },
            norm_cfg=dict(type='LN')
        )
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # 2. Build Hiera stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        
        for i in range(self.num_levels):
            level_dim = int(embed_dim * (2 ** i))
            prev_level_dim = embed_dim if i == 0 else int(embed_dim * (2 ** (i - 1)))
            level_heads = num_heads[i]
            
            downsample_norm = nn.Identity()
            downsample_conv = nn.Identity()
            if i > 0:
                downsample_norm = norm_layer(prev_level_dim)
                downsample_conv = nn.Conv2d(prev_level_dim, level_dim, kernel_size=2, stride=2)

            stage_blocks = nn.Sequential(*[
                HieraBlock(
                    dim=level_dim,
                    num_heads=level_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[sum(depths[:i]) + j],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for j in range(depths[i])
            ])
            
            self.levels.append(nn.ModuleDict({
                "downsample_norm": downsample_norm,
                "downsample_conv": downsample_conv,
                "blocks": stage_blocks,
            }))

    def init_weights(self, pretrained=None):
        if pretrained is None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.apply(self._init_weights)
        else:
            logger = get_root_logger()
            logger.info(f"Loading pretrained weights from {pretrained} is not implemented yet.")
            pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, x):
        """The main forward pass of the Hiera backbone with corrected dimension handling."""
        x = self.patch_embed(x) # x is [B, L, C]
        x = x + self.pos_embed
        
        outs = []
        
        B, L, C = x.shape
        H = W = int(L**0.5)

        for i, level in enumerate(self.levels):
            if i > 0:
                # 1. Apply LayerNorm on the 3D sequence tensor (e.g., [B, 784, 96])
                x = level["downsample_norm"](x)

                # 2. Reshape from 3D sequence to 4D image for convolution
                x = x.reshape(B, H, W, C).permute(0, 3, 1, 2) # -> [B, C, H, W]
                
                # 3. Apply convolution for downsampling
                x = level["downsample_conv"](x) # -> [B, C*2, H/2, W/2]
                
                # 4. Get new shape and reshape back to 3D sequence for Transformer blocks
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1).reshape(B, H * W, C) # -> [B, L_new, C_new]

            # 5. Apply HieraBlocks (operates on 3D sequence)
            x = level["blocks"](x)
            
            if i in self.out_indices:
                # Reshape output to 4D image format for the decoder
                out = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)
