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

# 使用从项目根目录开始的绝对导入路径，这是最稳健的方式
from models.GBC import GBC, BottConv
from models.PAF import PAF


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
        """
        生成蛇形扫描顺序 (Serpentine Scan Sequence)。
        """
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        o1_inverse, o2_inverse, o3_inverse, o4_inverse = [-1] * L, [-1] * L, [-1] * L, [-1] * L

        # --- 1. 水平蛇形扫描 --- #
        row, col = 0, 0
        for i in range(L):
            o1_inverse[row * W + col] = i
            o1.append(row * W + col)
            if row % 2 == 0:
                if col == W - 1: row += 1
                else: col += 1
            else:
                if col == 0: row += 1
                else: col -= 1

        # --- 2. 垂直蛇形扫描 --- #
        row, col = 0, 0
        for i in range(L):
            o2_inverse[row * W + col] = i
            o2.append(row * W + col)
            if col % 2 == 0:
                if row == H - 1: col += 1
                else: row += 1
            else:
                if row == 0: col += 1
                else: row -= 1

        # --- 3. 副对角线蛇形扫描 (左上 -> 右下) --- #
        diagonals = [[] for _ in range(H + W - 1)]
        for i in range(H):
            for j in range(W):
                diagonals[i + j].append(i * W + j)

        for k, diag in enumerate(diagonals):
            if k % 2 == 1: diag.reverse()
            for idx in diag:
                o3_inverse[idx] = len(o3)
                o3.append(idx)

        # --- 4. 主对角线蛇形扫描 (右上 -> 左下) --- #
        diagonals = [[] for _ in range(H + W - 1)]
        for i in range(H):
            for j in range(W):
                diagonals[i - j + (W - 1)].append(i * W + j)

        for k, diag in enumerate(diagonals):
            if k % 2 == 1: diag.reverse()
            for idx in diag:
                o4_inverse[idx] = len(o4)
                o4.append(idx)

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
               (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse))

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
        B = B.permute(0, 2, 1).contiguous() # Shape: (B, d_state, L)
        C = C.permute(0, 2, 1).contiguous()

        # 获取扫描顺序
        orders, inverse_orders = self.sass(hw_shape)
        
        # [FIXED] 正确地为每个扫描方向创建可广播的 delta_B 张量
        delta_Bs = []
        # 假设水平、垂直、两个对角线分别使用 1, 2, 0, 0 索引
        dir_indices = [1, 2, 0, 0] 
        for i in range(self.n_directions):
            # 从方向参数中获取 1D 向量, shape: (d_state,)
            d_vec = self.direction_Bs[dir_indices[i], :]
            # 将其整形为 (1, d_state, 1) 以便与 B (B, d_state, L) 进行广播相加
            dB = d_vec.unsqueeze(0).unsqueeze(-1)
            delta_Bs.append(dB)

        # 执行选择性扫描
        y_scan = []
        for i in range(self.n_directions):
            scan_order = orders[i]
            inv_order = inverse_orders[i]
            delta_B = delta_Bs[i]

            # 对输入序列重新排序
            x_reordered = x_conv[:, scan_order, :].permute(0, 2, 1).contiguous()

            # 执行扫描
            y = selective_scan_fn(
                x_reordered, dt, A, (B + delta_B).contiguous(), C, self.D.float(),
                z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False
            )

            # 恢复原始顺序
            y_unordered = y.permute(0, 2, 1)
            y_reconstructed = y_unordered[:, inv_order, :]
            y_scan.append(y_reconstructed)

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
