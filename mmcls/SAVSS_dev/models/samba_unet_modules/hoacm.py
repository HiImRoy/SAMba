# Author: Roy
# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 该文件根据 SAMba-UNet 论文第3.3节和图3进行了完全重构，以精确实现论文中描述的HOACM架构。

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidualMLP(nn.Module):
    """
    功能: 倒置残差MLP模块。
    对应论文图3中HOACM最后的处理模块，一个带有残差连接的倒置瓶颈MLP。
    """
    def __init__(self, dim, mlp_ratio=2.0, act_layer=nn.GELU, drop=0.0):
        """
        Args:
            dim (int): 输入输出维度。
            mlp_ratio (float): 中间扩展比例，默认为2.0。
            act_layer (nn.Module): 激活函数，默认为nn.GELU。
            drop (float): Dropout比率，默认为0.0。
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)  # 计算隐藏层维度
        # 1x1卷积，用于升维
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.act = act_layer()  # 激活函数
        # 3x3深度可分离卷积
        self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        # 1x1卷积，用于降维
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播函数。
        数据流: shortcut -> conv1 -> act -> conv_dw -> act -> conv2 -> drop -> output + shortcut
        """
        shortcut = x  # 保存残差连接的输入
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv_dw(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.drop(x)
        return shortcut + x  # 应用残差连接


class BifurcatedSelectiveEmphasisAttention(nn.Module):
    """
    功能: 分叉选择性强调注意力 (BSEA)。
    对应论文3.3节中的BSEA，专门处理Mamba/SAVSS分支的特征，
    通过分叉的自注意力机制捕捉全局和局部上下文。
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): 输入特征维度。
        """
        super().__init__()
        # 定义三个共享权重的1x1卷积层，用于生成Q, K, V
        self.conv_q = nn.Conv2d(dim, dim, 1)
        self.conv_k = nn.Conv2d(dim, dim, 1)
        self.conv_v = nn.Conv2d(dim, dim, 1)

    def _self_attention_path(self, pooled_features):
        """一个辅助函数，用于执行共享的自注意力计算。"""
        B, C, H_p, W_p = pooled_features.shape
        # 从池化后的特征生成 Q, K, V
        q = self.conv_q(pooled_features).flatten(2)  # (B, C, N)
        k = self.conv_k(pooled_features).flatten(2)  # (B, C, N)
        v = self.conv_v(pooled_features).flatten(2)  # (B, C, N)

        # 计算相似度矩阵 (B, N, C) @ (B, C, N) -> (B, N, N)
        attn_matrix = F.softmax(torch.bmm(q.transpose(-2, -1), k), dim=-1)

        # 聚合上下文信息 (B, C, N) @ (B, N, N) -> (B, C, N)
        attended_v = torch.bmm(v, attn_matrix.transpose(-2, -1))
        
        # 重塑回原始池化特征的图像形状
        return attended_v.reshape(B, C, H_p, W_p)

    def forward(self, x_mamba):
        """
        BSEA的前向传播。
        Args:
            x_mamba (torch.Tensor): 来自Mamba/SAVSS分支的特征图。
        """
        # 1. 分叉池化 (Bifurcated Pooling)
        # 使用自适应池化将特征图缩小一半，以捕捉不同尺度的上下文
        pool_size = (x_mamba.shape[2] // 2, x_mamba.shape[3] // 2)
        x_avg = F.adaptive_avg_pool2d(x_mamba, pool_size)
        x_max = F.adaptive_max_pool2d(x_mamba, pool_size)

        # 2. 共享自注意力 (Shared Self-Attention)
        x_prime_avg = self._self_attention_path(x_avg)
        x_prime_max = self._self_attention_path(x_max)

        # 3. 融合与增强 (Fusion and Enhancement)
        # 逐元素相加，并上采样回原始尺寸
        fused_attn_map = F.interpolate(x_prime_avg + x_prime_max, size=x_mamba.shape[2:], mode='bilinear', align_corners=False)
        
        # 使用Sigmoid生成门控信号，并与原始输入逐元素相乘
        enhanced_x = x_mamba * torch.sigmoid(fused_attn_map)
        return enhanced_x


class OmniscientContextualAttention(nn.Module):
    """
    功能: 全知上下文注意力 (OCA)。
    对应论文3.3节中的OCA，专门处理SAM/Hiera分支的特征，
    通过全局上下文和门控空间注意力增强像素级语义。
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): 输入特征维度。
        """
        super().__init__()
        # 门控空间注意力 (GSA) 卷积
        self.gsa_conv = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        
        # 最终重校准卷积 (final_conv)，使用轻量化的深度可分离卷积实现
        self.final_conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3, groups=2, bias=False),
            # 逐点卷积
            nn.Conv2d(in_channels=2, out_channels=dim, kernel_size=1, bias=False)
        )

    def forward(self, x_sam):
        """
        OCA的前向传播。
        Args:
            x_sam (torch.Tensor): 来自SAM/Hiera分支的特征图。
        """
        # 1. 双通道压缩 (Dual-Channel Compression)
        x_msam, _ = torch.max(x_sam, dim=1, keepdim=True)  # 通道最大池化
        x_asam = torch.mean(x_sam, dim=1, keepdim=True)  # 通道平均池化

        # 2. 拼接 (Concatenation)
        x_cat = torch.cat([x_msam, x_asam], dim=1)  # Shape: (B, 2, H, W)

        # 3. 全局上下文感知注意力 (GCAA)
        x_global = F.adaptive_avg_pool2d(x_cat, (1, 1)) * x_cat

        # 4. 门控空间注意力 (GSA)
        x_gsa = x_global * torch.sigmoid(self.gsa_conv(x_global))

        # 5. 最终重校准 (Final Recalibration)
        final_gate = torch.sigmoid(self.final_conv(x_gsa))
        x_prime_sam = x_sam * final_gate

        return x_prime_sam


class HOACM(nn.Module):
    """
    功能: 异构全注意力融合模块 (HOACM)。
    作为顶层容器，协调OCA和BSEA，并完成最终的特征融合，精确对应论文图3的架构。
    """
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        """
        Args:
            dim (int): 输入输出维度。
            mlp_ratio (float): 传递给InvertedResidualMLP的扩展比例。
            drop (float): Dropout比率。
        """
        super().__init__()
        self.oca = OmniscientContextualAttention(dim)
        self.bsea = BifurcatedSelectiveEmphasisAttention(dim)
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1)  # 1x1卷积用于融合
        self.mlp = InvertedResidualMLP(dim, mlp_ratio, drop=drop)
        self.norm = nn.LayerNorm(dim)  # 最终的层归一化

    def forward(self, x_sam, x_mamba):
        """
        HOACM的前向传播。
        Args:
            x_sam (torch.Tensor): 来自SAM/Hiera分支的特征 (B, C, H, W)。
            x_mamba (torch.Tensor): 来自Mamba/SAVSS分支的特征 (B, C, H, W)。
        """
        # 1. 使用各自特定的注意力模块处理每个流
        sam_enhanced = self.oca(x_sam)
        mamba_enhanced = self.bsea(x_mamba)

        # 2. 融合处理后的特征
        fused = torch.cat([sam_enhanced, mamba_enhanced], dim=1)
        fused = self.fusion_conv(fused)

        # 3. 通过最后的MLP块
        output = self.mlp(fused)

        # 4. 最终的归一化 (Permute -> LayerNorm -> Permute)
        # LayerNorm需要 (..., C) 格式的输入
        output = output.permute(0, 2, 3, 1)  # (B, H, W, C)
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)  # (B, C, H, W)

        return output
