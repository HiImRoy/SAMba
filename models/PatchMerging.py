# Author: Roy
# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    """
    Patch Merging Layer for Downsampling.
    This implementation is based on the official Swin Transformer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x (torch.Tensor): Input feature, tensor size (B, H*W, C).
            H (int): Height of input feature map.
            W (int): Width of input feature map.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # Reshape to 2D
        x = x.view(B, H, W, C)

        # Partition and Group
        x0 = x[:, 0::2, 0::2, :]  # Top-Left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-Left
        x2 = x[:, 0::2, 1::2, :]  # Top-Right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-Right

        # Concatenate
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # (B, H*W/4, 4*C)

        # Normalize and Project
        x = self.norm(x)
        x = self.reduction(x)  # (B, H*W/4, 2*C)

        # Calculate new H, W
        H_new, W_new = H // 2, W // 2

        return x, H_new, W_new