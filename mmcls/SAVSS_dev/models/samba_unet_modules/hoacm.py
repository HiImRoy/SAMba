# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 该文件根据 SAMba-UNet 论文中的详细描述进行了修订。

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidualMLP(nn.Module):
    """倒置残差 MLP 块，用作 HOACM 的最终处理步骤。"""
    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio) # 隐藏层维度
        # 1x1 卷积，用于升维
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.act = act_layer() # 激活函数
        # 深度可分离卷积
        self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        # 1x1 卷积，用于降维
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x # 保存残差连接的输入
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv_dw(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.drop(x)
        return shortcut + x # 应用残差连接


class BifurcatedSelectiveEmphasisAttention(nn.Module):
    """分叉选择性强调注意力 (BSEA)。

    根据 SAMba-UNet 论文的 3.3 节和公式 (5)-(12) 实现。
    它主要处理来自 Mamba/SAVSS 编码器的特征。
    """
    def __init__(self, dim):
        super().__init__()
        # 共享权重的自适应自注意力机制 (公式 7-9)
        self.conv_q = nn.Conv2d(dim, dim, 1)
        self.conv_k = nn.Conv2d(dim, dim, 1)
        self.conv_v = nn.Conv2d(dim, dim, 1)

    def _self_attention_path(self, pooled_features):
        # 从池化后的特征生成 Q, K, V
        q = self.conv_q(pooled_features).flatten(2) # (B, C, N)
        k = self.conv_k(pooled_features).flatten(2) # (B, C, N)
        v = self.conv_v(pooled_features).flatten(2) # (B, C, N)

        # 计算相似度矩阵 (公式 10)
        # 假设 Q 和 K 需要转置以进行矩阵乘法 (B, N, C)
        attn_matrix = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)

        # 聚合上下文信息 (公式 11)
        # 结果形状为 (B, N, C)，需要重塑
        attended_v = torch.bmm(v, attn_matrix.transpose(1, 2))
        
        # 重塑回原始池化特征的形状
        H_p, W_p = pooled_features.shape[2:]
        return attended_v.transpose(1, 2).reshape(pooled_features.shape)

    def forward(self, x_mamba):
        # 双路径池化 (公式 5, 6)
        # 论文对池化大小的描述比较模糊，我们假设使用自适应池化到一个较小的网格
        x_avg = F.adaptive_avg_pool2d(x_mamba, (x_mamba.shape[2]//2, x_mamba.shape[3]//2))
        x_max = F.adaptive_max_pool2d(x_mamba, (x_mamba.shape[2]//2, x_mamba.shape[3]//2))

        # 通过共享权重的自注意力机制处理两个路径 (公式 7-11)
        x_prime_avg = self._self_attention_path(x_avg)
        x_prime_max = self._self_attention_path(x_max)

        # 融合并创建空间注意力图 (解读公式 12)
        # 将注意力图上采样以匹配原始特征尺寸
        fused_attn_map = F.interpolate(x_prime_avg + x_prime_max, size=x_mamba.shape[2:], mode='bilinear')
        
        # 通过逐元素相乘来增强原始 Mamba 特征
        # 公式 12 中的 Softmax 似乎不太合适，使用 Sigmoid 进行门控更为常见。
        enhanced_x = x_mamba * torch.sigmoid(fused_attn_map)
        return enhanced_x


class OmniscientContextualAttention(nn.Module):
    """全知上下文注意力 (OCA)。

    根据 SAMba-UNet 论文的 3.3 节和公式 (13)-(17) 实现。
    它主要增强来自 SAM2/Hiera 编码器的特征。
    """
    def __init__(self, dim):
        super().__init__()
        # 门控空间注意力 (GSA) 卷积 (公式 17)
        self.gsa_conv = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        # 最终注意力图卷积 (公式 17, 最后部分)
        self.final_conv = nn.Conv2d(2, dim, kernel_size=7, padding=3)

    def forward(self, x_sam):
        # 双通道压缩 (公式 13, 14)
        x_msam, _ = torch.max(x_sam, dim=1, keepdim=True) # 最大值池化
        x_asam = torch.mean(x_sam, dim=1, keepdim=True) # 平均值池化

        # 沿通道维度拼接 (公式 15)
        x_cat = torch.cat([x_msam, x_asam], dim=1) # Shape: (B, 2, H, W)

        # 全局特征表示与对比度增强 (公式 16)
        x_global_pooled = F.adaptive_avg_pool2d(x_cat, (1, 1))
        x_global = x_global_pooled * x_cat # 广播乘法

        # 门控空间注意力 (GSA) 机制 (公式 17, 第一部分)
        gsa_gate = torch.sigmoid(self.gsa_conv(x_global))
        x_gsa = x_global * gsa_gate

        # 生成最终空间注意力权重并重新校准 SAM2 特征 (公式 17, 第二部分)
        final_gate = torch.sigmoid(self.final_conv(x_gsa))
        x_prime_sam = x_sam * final_gate

        return x_prime_sam


class HOACM(nn.Module):
    """异构全注意力融合模块 (HOACM)。

    该模块使用修订后的 OCA 和 BSEA 子模块，
    融合来自 SAM (Hiera) 和 Mamba (SAVSS) 编码器的特征。
    """
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.oca = OmniscientContextualAttention(dim) # 全知上下文注意力
        self.bsea = BifurcatedSelectiveEmphasisAttention(dim) # 分叉选择性强调注意力
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1) # 用于合并拼接后的特征
        self.mlp = InvertedResidualMLP(dim, mlp_ratio, drop=drop) # 倒置残差 MLP
        self.norm = nn.LayerNorm(dim) # 层归一化

    def forward(self, x_sam, x_mamba):
        """
        Args:
            x_sam (torch.Tensor): 来自 SAM/Hiera 编码器的特征 (B, C, H, W)。
            x_mamba (torch.Tensor): 来自 Mamba/SAVSS 编码器的特征 (B, C, H, W)。
        """
        # 使用各自特定的、修订后的注意力模块处理每个流
        sam_enhanced = self.oca(x_sam)
        mamba_enhanced = self.bsea(x_mamba)

        # 通过拼接和 1x1 卷积来融合处理后的特征
        fused = torch.cat([sam_enhanced, mamba_enhanced], dim=1)
        fused = self.fusion_conv(fused)

        # 通过最后的 MLP 块
        output = self.mlp(fused)

        # 最终的归一化
        output = output.permute(0, 2, 3, 1)  # (B, H, W, C)
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)  # (B, C, H, W)

        return output
