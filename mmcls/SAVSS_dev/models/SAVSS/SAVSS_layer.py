# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# **重构**: 将 GBC 模块中的 BatchNorm2d 替换为 GroupNorm，以提高小批量训练的稳定性。

import math
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm

# --- 内联 GBC.py 内容 (已修改为使用 GroupNorm) ---
class BottConv(nn.Module):
    """瓶颈卷积块 (Bottleneck Convolution)。
    
    这是一个标准的2D卷积瓶颈模块，常用于ResNet等架构中。
    它通过 1x1卷积 -> 3x3卷积 -> 1x1卷积 的结构，在减少计算量的同时提取空间特征。
    【已修改】: 将所有 BatchNorm2d 替换为 GroupNorm。
    """
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, padding, stride, num_groups=8):
        super(BottConv, self).__init__()
        self.bott_conv = nn.Sequential(
            # 1. 压缩通道: 1x1卷积，减少通道数，形成瓶颈
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            # 2. 空间卷积: 在瓶颈上进行3x3卷积，提取空间特征
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            # 3. 扩展通道: 1x1卷积，恢复通道数
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 输入必须是2D特征图，形如 [B, C, H, W]
        return self.bott_conv(x)

class GBC(nn.Module):
    """全局瓶颈卷积块 (Global Bottleneck Convolution)。
    
    这是一个包裹了BottConv并带有残差连接的2D卷积模块。
    【已修改】: 将 BatchNorm2d 替换为 GroupNorm。
    """
    def __init__(self, in_channels, out_channels=None, intermediate_channels=None, num_groups=8):
        super(GBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.intermediate_channels = intermediate_channels if intermediate_channels is not None else in_channels // 4

        self.gwc = BottConv(self.in_channels, self.out_channels, self.intermediate_channels, 3, 1, 1, num_groups=num_groups)
        self.norm = nn.GroupNorm(num_groups, self.out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # 输入是2D特征图 [B, C, H, W]
        res = x
        x = self.gwc(x)
        x = self.act(self.norm(x))
        # 返回经过卷积和残差连接后的结果
        return x + res

# --- 内联 PAF.py 内容并修复维度问题 ---
class PAF(nn.Module):
    """金字塔注意力融合模块 (Pyramid Attention Fusion)。
    
    该模块接收两个2D特征图，利用一个特征图(x_s)生成空间注意力，
    然后将该注意力应用到另一个特征图(x_v)上，实现特征的动态融合。
    """
    def __init__(self, in_channels, hidden_dim, kernel_size=3, padding=1, stride=1):
        super(PAF, self).__init__()
        self.conv_s = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv_spatial = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x_v, x_s):
        # 两个输入都必须是2D特征图 [B, C, H, W]
        # 1. 将两个输入都投影到相同的隐藏维度
        x_s_proj = self.conv_s(x_s)
        x_v_proj = self.conv_v(x_v)
        
        # 2. 利用 x_s 分支生成空间注意力图
        x_s_proj = self.gelu(x_s_proj)
        attn = self.conv_spatial(x_s_proj)
        attn = self.sigmoid(attn)
        
        # 3. 将注意力图应用到 x_v 分支上
        x_att = x_v_proj * attn
        
        # 4. 将融合后的特征投影回原始输入维度
        return self.conv_out(x_att)

# --- SAVSS_2D 核心实现 ---
class SAVSS_2D(nn.Module):
    """ 2D选择性扫描状态空间模型 (Selective Scan State Space Model) 的核心实现。
    
    这个模块是Mamba在2D图像上的应用。它接收1D序列作为输入，但内部会通过一个2D卷积
    来增强局部信息，然后再执行核心的选择性扫描操作。
    【已修改】: 内部的 BottConv 也已更新为使用 GroupNorm。
    """
    def __init__(
            self,
            d_model, # 模型维度 (输入/输出通道数)
            d_state=16, # 状态维度 N
            expand=2, # 扩展因子 E
            dt_rank="auto", # delta_t 的秩
            conv_size=7, # 内部卷积核大小
            num_groups=8, # 为GroupNorm设置组数
            **kwargs # 接收其他Mamba参数
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) # 内部维度 E*D
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.n_directions = 2 # 简化的扫描方向数 (水平和垂直)

        # 输入投影层: 从 D -> 2*E*D，用于生成 x 和 z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=kwargs.get('bias', False))

        assert conv_size % 2 == 1
        # 内部2D卷积层: 在Mamba核心操作前，用于增强局部上下文信息
        self.conv2d = BottConv(
            in_channels=self.d_inner, 
            out_channels=self.d_inner, 
            mid_channels=self.d_inner // 16, 
            kernel_size=3, padding=1, stride=1, 
            num_groups=num_groups
        )
        self.act = nn.SiLU()

        # 动态参数生成层: 从 E*D -> dt_rank + 2*d_state，用于生成 dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        # dt 的投影层: 从 dt_rank -> E*D
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 dt_proj 的权重和偏置，这是Mamba的关键技巧
        dt_init_std = self.dt_rank ** -0.5 * kwargs.get('dt_scale', 1.0)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(kwargs.get('dt_max', 0.1)) - math.log(kwargs.get('dt_min', 0.001))) + math.log(kwargs.get('dt_min', 0.001)))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad(): self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # 初始化状态矩阵 A (离散化后) 和旁路矩阵 D
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A)) # A 是通过学习 A_log 的指数得到的
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # 输出投影层: 从 E*D -> D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=kwargs.get('bias', False))
        
        # 方向相关的 B 矩阵参数 (为每个扫描方向学习一个独立的B偏置)
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions * 2 + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    def sass(self, hw_shape):
        """生成蛇形扫描顺序 (Serpentine Scan Sequence)。为2D图像生成高效的扫描路径。"""
        H, W = hw_shape
        L = H * W
        o1, o2 = [], []
        d1, d2 = [], []
        o1_inverse, o2_inverse = [-1] * L, [-1] * L

        # 1. 水平蛇形扫描 (从右下角或左下角开始，取决于高度奇偶性)
        i, j = (H - 1, W - 1) if H % 2 == 1 else (H - 1, 0)
        j_d = "left" if H % 2 == 1 else "right"
        while i > -1:
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1: j += 1; d1.append(1)
                else: i -= 1; d1.append(3); j_d = "left"
            else:
                if j > 0: j -= 1; d1.append(2)
                else: i -= 1; d1.append(3); j_d = "right"
        d1 = [0] + d1[:-1]

        # 2. 垂直蛇形扫描 (从左上角开始)
        i, j = 0, 0
        i_d = "down"
        while j < W:
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1: i += 1; d2.append(4)
                else: j += 1; d2.append(1); i_d = "up"
            else:
                if i > 0: i -= 1; d2.append(3)
                else: j += 1; d2.append(1); i_d = "down"
        d2 = [0] + d2[:-1]

        return (tuple(o1), tuple(o2)), (tuple(o1_inverse), tuple(o2_inverse)), (tuple(d1), tuple(d2))

    def forward(self, x, hw_shape):
        # --- 1. 输入投影 ---
        # 输入 x 是 1D 序列, Shape: [B, L, D]
        batch_size, L, D = x.shape
        H, W = hw_shape
        E = self.d_inner

        xz = self.in_proj(x) # Shape: [B, L, 2*E]
        x, z = xz.chunk(2, dim=-1) # x.shape: [B, L, E], z.shape: [B, L, E]

        # --- 2. 内部2D卷积路径 (增强局部性) ---
        # a. 将 1D 序列 x 变形为 2D 特征图
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2).contiguous() # Shape: [B, E, H, W]
        # b. 应用2D卷积
        x_2d = self.act(self.conv2d(x_2d)) # Shape: [B, E, H, W]
        # c. 将卷积后的 2D 特征图变形回 1D 序列
        x_conv = x_2d.permute(0, 2, 3, 1).contiguous().reshape(batch_size, L, E) # Shape: [B, L, E]

        # --- 3. 计算选择性扫描的动态参数 (dt, B, C) ---
        A = -torch.exp(self.A_log.float()) # 固定的 A 矩阵, Shape: [E, N]
        x_dbl = self.x_proj(x_conv) # Shape: [B, L, dt_rank + 2*N]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1).contiguous() # Shape: [B, E, L]
        B = B.permute(0, 2, 1).contiguous() # Shape: [B, N, L]
        C = C.permute(0, 2, 1).contiguous() # Shape: [B, N, L]

        # --- 4. 执行多方向选择性扫描 ---
        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]

        y_scan = []
        for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs):
            # a. 根据扫描顺序(o)重排输入序列
            x_scan = x_conv[:, o, :].permute(0, 2, 1).contiguous() # Shape: [B, E, L]
            # b. 执行核心扫描操作
            y = selective_scan_fn(
                x_scan, dt, A, (B + dB).contiguous(), C, self.D.float(),
                z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False
            ).permute(0, 2, 1) # Shape: [B, L, E]
            # c. 根据逆序(inv_order)恢复原始顺序
            y_scan.append(y[:, inv_order, :])

        # --- 5. 融合与输出 ---
        # a. 融合多个扫描方向的结果
        y = sum(y_scan)
        # b. 应用门控机制 (gating)
        y = y * self.act(z)
        # c. 输出投影
        out = self.out_proj(y) # Shape: [B, L, D]

        return out

class SAVSS_Layer(nn.Module):
    """SAVSS 层，是 Mamba 编码器的基本构建块。
    
    这个模块整合了 GBC (2D卷积), SAVSS_2D (1D序列处理), 和 PAF (2D融合) 
    以及残差连接，形成一个强大的混合模型层。
    """
    def __init__(self, embed_dims, use_rms_norm, with_dwconv, drop_path_rate, mamba_cfg):
        super(SAVSS_Layer, self).__init__()
        
        # 为 SAVSS_2D 模块准备参数，并传入 num_groups=8
        mamba_cfg.update({'d_model': embed_dims, 'num_groups': 8})
        
        # 归一化层
        self.norm = RMSNorm(embed_dims) if use_rms_norm else nn.LayerNorm(embed_dims)
        
        # 核心 Mamba 模块
        self.SAVSS_2D = SAVSS_2D(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        
        # 2D 卷积模块 (num_groups=8)
        self.GBC_C = GBC(embed_dims, num_groups=8)
        # 2D 融合模块
        self.PAF = PAF(embed_dims, embed_dims // 2)

    def forward(self, x, hw_shape):
        # --- 初始状态 ---
        # 输入 x 是一个 1D 序列 (Token)
        # Shape: [B, L, C] (例如: [4, 12544, 64])
        B, L, C = x.shape
        H, W = hw_shape
        
        # --- 并行分支 1: 原始输入残差 ---
        # 直接保留原始输入，用于最终的相加融合
        x_token_res = x
        
        # --- 数据准备: 将 1D 序列转换为 2D 特征图，为 GBC 和 PAF 做准备 ---
        x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() # Shape: [B, C, H, W]

        # --- 并行分支 2: GBC -> Mamba 路径 ---
        # 1. GBC 路径 (2D 卷积): 提取局部空间特征
        x_2d_gbc = self.GBC_C(x_2d) # Shape: [B, C, H, W]
        # 2. 变形 (2D -> 1D): 将 GBC 的输出压平为 1D 序列，为 Mamba 做准备
        x_gbc_token = x_2d_gbc.permute(0, 2, 3, 1).reshape(B, L, C) # Shape: [B, L, C]
        # 3. Mamba 路径 (1D 序列处理): 提取长距离依赖
        mamba_out_token = self.drop_path(self.SAVSS_2D(self.norm(x_gbc_token), hw_shape)) # Shape: [B, L, C]
        
        # --- 并行分支 3: PAF 融合路径 ---
        # 1. 变形 (1D -> 2D): 将 Mamba 的输出展开为 2D 特征图，为 PAF 做准备
        mamba_out_2d = mamba_out_token.permute(0, 2, 1).reshape(B, C, H, W) # Shape: [B, C, H, W]
        # 2. PAF 融合 (2D 融合): 融合 Mamba 的输出 (mamba_out_2d) 和 原始特征 (x_2d)
        paf_out_2d = self.PAF(x_2d, mamba_out_2d) # Shape: [B, C, H, W]
        # 3. 变形 (2D -> 1D): 将 PAF 的输出压平为 1D 序列，为最终融合做准备
        paf_out_token = paf_out_2d.permute(0, 2, 3, 1).reshape(B, L, C) # Shape: [B, L, C]

        # --- 最终融合 ---
        # 将三个并行分支的结果相加
        # 1. 原始输入残差 (x_token_res)
        # 2. GBC->Mamba路径的输出 (mamba_out_token)
        # 3. PAF融合路径的输出 (paf_out_token)
        final_out_token = x_token_res + mamba_out_token + paf_out_token

        return final_out_token
