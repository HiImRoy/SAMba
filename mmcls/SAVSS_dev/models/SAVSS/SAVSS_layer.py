# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm

# --- 内联 GBC.py 内容以解决循环导入问题 ---
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

# --- 内联 PAF.py 内容并修复维度问题 ---
class PAF(nn.Module):
    """金字塔注意力融合模块 (Pyramid Attention Fusion)。"""
    def __init__(self, in_channels, hidden_dim, kernel_size=3, padding=1, stride=1):
        super(PAF, self).__init__()
        self.conv_s = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, hidden_dim, kernel_size=1) # 将 x_v 投影到 hidden_dim
        self.conv_spatial = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim)
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1) # 投影回 in_channels
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x_v, x_s):
        # 将两个输入都投影到相同的隐藏维度
        x_s_proj = self.conv_s(x_s)
        x_v_proj = self.conv_v(x_v)
        
        x_s_proj = self.gelu(x_s_proj)
        attn = self.conv_spatial(x_s_proj) # 生成空间注意力图
        attn = self.sigmoid(attn)
        
        # 在隐藏维度上进行乘法
        x_att = x_v_proj * attn
        
        # 投影回原始输入维度
        return self.conv_out(x_att)

# --- SAVSS_Layer 原始内容 (含修复) ---
class SAVSS_2D(nn.Module):
    """ 2D选择性扫描状态空间模型 (Selective Scan State Space Model) 的核心实现。"""
    def __init__(
            self,
            d_model, # 模型维度
            d_state=16, # 状态维度
            expand=2, # 扩展因子
            dt_rank="auto", # delta t 的秩
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=7, # 卷积核大小
            bias=False,
            conv_bias=False,
            init_layer_scale=None,
            default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) # 内部维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.n_directions = 4 # 扫描方向数

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        # 输入投影层
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        # 2D 卷积层，这里使用了瓶颈卷积
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16, kernel_size=3, padding=1, stride=1)
        self.act = nn.SiLU()

        # x 到 dt, B, C 的投影层
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        # dt 的投影层
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 dt_proj 的权重和偏置
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # 初始化状态矩阵 A 和 D
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        # 输出投影层
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        # 方向相关的 B 矩阵参数
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    def sass(self, hw_shape):
        """生成蛇形扫描顺序 (Serpentine Scan Sequence)。"""
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse, o2_inverse, o3_inverse, o4_inverse = [-1] * L, [-1] * L, [-1] * L, [-1] * L

        # 水平蛇形扫描
        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

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

        # 垂直蛇形扫描
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
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        xz = self.in_proj(x)
        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=-1)
        # 2D 卷积路径
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        # 计算 SSM 的参数 dt, B, C
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        # 获取扫描顺序和方向
        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]

        # 执行选择性扫描
        y_scan = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(), dt, A, (B + dB).contiguous(), C, self.D.float(),
                z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False
            ).permute(0, 2, 1)[:, inv_order, :]
            for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs)
        ]

        # 融合多个扫描方向的结果
        y = sum(y_scan) * self.act(z)
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out

class SAVSS_Layer(nn.Module):
    """SAVSS 层，是 Mamba 编码器的基本构建块。"""
    def __init__(self, embed_dims, use_rms_norm, with_dwconv, drop_path_rate, mamba_cfg):
        super(SAVSS_Layer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})
        # 选择归一化方式
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        # 是否使用深度可分离卷积
        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False, groups=embed_dims),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )

        self.SAVSS_2D = SAVSS_2D(**mamba_cfg) # 核心 Mamba 模块
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        
        num_groups = 16 if embed_dims % 16 == 0 else embed_dims % 8 if embed_dims % 8 == 0 else embed_dims % 4 if embed_dims % 4 == 0 else 2
        self.GN = nn.GroupNorm(num_channels=embed_dims, num_groups=num_groups)
        self.linear = nn.Linear(in_features=embed_dims, out_features=embed_dims, bias=True)
        self.GBC_C = GBC(embed_dims) # 全局瓶颈卷积
        self.PAF = PAF(embed_dims, embed_dims // 2) # 金字塔注意力融合

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        
        x_token_res = x # 保存原始输入的残差连接
        x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # GBC 路径
        x_2d_gbc = self.GBC_C(x_2d)
        x_gbc_token = x_2d_gbc.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Mamba 路径
        mamba_out_token = self.drop_path(self.SAVSS_2D(self.norm(x_gbc_token), hw_shape))
        mamba_out_2d = mamba_out_token.permute(0, 2, 1).reshape(B, C, H, W)
        
        # PAF 融合原始特征 (x_2d) 和 mamba 输出 (mamba_out_2d)
        paf_out_2d = self.PAF(x_2d, mamba_out_2d)

        # 简化后的残差连接路径
        final_out_token = x_token_res + mamba_out_token + paf_out_2d.permute(0, 2, 3, 1).reshape(B, L, C)

        return final_out_token
