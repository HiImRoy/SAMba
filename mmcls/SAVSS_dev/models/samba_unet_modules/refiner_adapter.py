# Copyright (c) Your Company. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is revised based on the detailed description in the SAMba-UNet paper.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAdapter(nn.Module):
    """A simple MLP adapter module.

    This module is used to enhance the non-linear capabilities of the features.
    The architecture is a simple two-layer MLP with a GeLU activation in between.
    It is used in parallel with the DynamicFeatureFusionRefiner.
    """

    def __init__(self, dim, mlp_ratio=2.0, drop=0.1):
        """
        Args:
            dim (int): Input and output dimension.
            mlp_ratio (float): The ratio to determine the hidden dimension size.
            drop (float): Dropout rate.
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass for the MLP adapter.
        Input shape: (B, C, H, W)
        The MLP operates on the channel dimension, so we permute.
        """
        # Permute to (B, H, W, C) for Linear layer
        x_permuted = x.permute(0, 2, 3, 1)
        
        out = self.fc1(x_permuted)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        
        # Permute back to (B, C, H, W)
        out = out.permute(0, 3, 1, 2)
        return out


class DynamicFeatureFusionRefiner(nn.Module):
    """Dynamic Feature Fusion Refiner (DFFR).

    This module is implemented strictly following the logic described in the
    SAMba-UNet paper (Section 3.2, Figure 2).
    It refines features to bridge the domain gap for medical images.

    The process is as follows:
    1. Dual Pooling (Avg + Max) to capture global context.
    2. Channel Attention to re-weight features based on global context (Eq. 1, 2).
    3. Spatial Enhancement using a cascaded Conv -> ReLU -> DeConv -> ReLU path (Eq. 3).
    4. Residual connection with the original input (Eq. 4).
    5. Layer Normalization for training stability.
    """

    def __init__(self, dim, reduction_ratio=0.25, spatial_kernel_size=3):
        """
        Args:
            dim (int): The number of channels in the input feature map (C).
            reduction_ratio (float): The bottleneck compression ratio `r` for channel attention.
            spatial_kernel_size (int): The kernel size for the spatial enhancement convs.
        """
        super().__init__()
        self.dim = dim
        bottleneck_dim = int(dim * reduction_ratio) # h in the paper
        padding = spatial_kernel_size // 2

        # --- Channel Attention components (Eq. 2) --- #
        self.channel_mlp_w1 = nn.Linear(dim * 2, bottleneck_dim)
        self.channel_mlp_w2 = nn.Linear(bottleneck_dim, dim)

        # --- Spatial Enhancement components (Eq. 3) --- #
        self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=spatial_kernel_size, padding=padding)
        self.spatial_deconv = nn.ConvTranspose2d(dim, dim, kernel_size=spatial_kernel_size, padding=padding)

        # --- Final LayerNorm (Eq. 4) --- #
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass for the DFFR module.
        Input shape: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.dim, "Input channel dimension must match module dimension."
        residual = x

        # --- 1. Dual Pooling & 2. Channel Attention (Eq. 1, 2) --- #
        # Dual adaptive pooling
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(B, C)
        
        # Concatenate along the channel dimension
        x_cat = torch.cat([avg_pool, max_pool], dim=1) # Shape: (B, 2*C)
        
        # Channel-wise dynamic calibration via learnable gating
        # Interpreting Eq.2 as a standard SE-like attention mechanism
        channel_weights = self.channel_mlp_w2(F.relu(self.channel_mlp_w1(x_cat)))
        channel_weights = torch.sigmoid(channel_weights).view(B, C, 1, 1)
        
        # Apply channel attention to the original feature map
        x_attn = x * channel_weights

        # --- 3. Spatial Enhancement (Eq. 3) --- #
        # Cascaded convolution path to augment local receptive fields
        x_sp = F.relu(self.spatial_deconv(F.relu(self.spatial_conv(x_attn))))

        # --- 4. Residual Connection & 5. Normalization (Eq. 4) --- #
        # Interpreting Eq.4 as a residual connection with the original input `x`
        refined_x = residual + x_sp

        # LayerNorm requires (B, ..., C) format
        refined_x = refined_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        refined_x = self.norm(refined_x)
        refined_x = refined_x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return refined_x
