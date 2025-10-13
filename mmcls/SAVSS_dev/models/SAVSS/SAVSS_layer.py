# Copyright (c) Roy. All rights reserved.

import math
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm

# 使用从项目根目录开始的绝对导入路径
from models.GBC import GBC
from models.PAF import PAF


class SS2D(nn.Module):
    """ 
    2D选择性扫描状态空间模型 (Selective Scan State Space Model) 的核心实现。
    为了模块化和清晰性，从原 SAVSS_Layer 中独立出来。
    作者: Roy
    """
    def __init__(
            self,
            d_model, # 模型维度
            d_state=16, # 状态维度
            # [REMOVED] SS2D 不应负责扩展，它应该直接处理传入的维度
            # expand=2, 
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
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # [MODIFIED] d_inner 就是 d_model，因为扩展在外部处理
        self.d_inner = self.d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.n_directions = 4 # 扫描方向数

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

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
        
        # 方向相关的 B 矩阵参数
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions, self.d_state))
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
        # x 已经是经过 Linear 和 SiLU 激活后的结果
        batch_size, L, E = x.shape
        A = -torch.exp(self.A_log.float())

        # 计算 SSM 的参数 dt, B, C
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous() # Shape: (B, d_state, L)
        C = C.permute(0, 2, 1).contiguous()

        # 获取扫描顺序
        orders, inverse_orders = self.sass(hw_shape)
        
        delta_Bs = []
        for i in range(self.n_directions):
            d_vec = self.direction_Bs[i, :]
            dB = d_vec.unsqueeze(0).unsqueeze(-1)
            delta_Bs.append(dB)

        # 执行选择性扫描
        y_scan = []
        for i in range(self.n_directions):
            scan_order = orders[i]
            inv_order = inverse_orders[i]
            delta_B = delta_Bs[i]

            x_reordered = x[:, scan_order, :].permute(0, 2, 1).contiguous()

            y = selective_scan_fn(
                x_reordered, dt, A, (B + delta_B).contiguous(), C, self.D.float(),
                z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False
            )

            y_unordered = y.permute(0, 2, 1)
            y_reconstructed = y_unordered[:, inv_order, :]
            y_scan.append(y_reconstructed)

        # 融合多个扫描方向的结果
        y = sum(y_scan)

        return y


class SAVSS_Block(nn.Module):
    """
    SAVSS 块，严格遵循 SCSegamba 论文图2(b)的架构实现。
    作者: Roy
    """
    def __init__(self, embed_dims, mamba_cfg, drop_path_rate=0., num_gbc=2, paf_mid_channels=None):
        super().__init__()
        self.embed_dims = embed_dims
        d_inner = int(mamba_cfg.get('expand', 2) * embed_dims)

        # [MODIFIED] 创建 SS2D 的配置，并移除 'expand' 参数
        ss2d_cfg = mamba_cfg.copy()
        ss2d_cfg.pop('expand', None)

        # 1. GBC 预处理模块
        self.gbc_modules = nn.ModuleList(
            [GBC(embed_dims) for _ in range(num_gbc)]
        )

        # 2. 双分支门控前的 LayerNorm
        self.norm = nn.LayerNorm(embed_dims)

        # 3. 上路 (Mamba 路径)
        self.linear_ss2d = nn.Linear(embed_dims, d_inner)
        self.act_ss2d = nn.SiLU()
        self.ss2d = SS2D(d_model=d_inner, **ss2d_cfg)

        # 4. 下路 (门控路径)
        self.linear_gate = nn.Linear(embed_dims, d_inner)
        self.act_gate = nn.SiLU()

        # 5. 后处理路径
        self.linear_post_fusion = nn.Linear(d_inner, embed_dims)
        
        paf_mid_channels = paf_mid_channels or embed_dims // 2
        self.paf = PAF(embed_dims, hidden_dim=paf_mid_channels)
        
        self.group_norm = nn.GroupNorm(num_groups=16, num_channels=embed_dims)
        self.linear_post_paf = nn.Linear(embed_dims, embed_dims)

        # DropPath
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate)) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x_input, hw_shape):
        """
        严格遵循 prompt 描述的数据流。
        """
        B, L, C = x_input.shape
        H, W = hw_shape

        # 1. GBC 预处理
        x_2d = x_input.reshape(B, H, W, C).permute(0, 3, 1, 2)
        for gbc_module in self.gbc_modules:
            x_2d = gbc_module(x_2d)
        x_gbc = x_2d.permute(0, 2, 3, 1).reshape(B, L, C)

        # 2. 双分支门控
        x_norm = self.norm(x_gbc)

        # 上路
        ss2d_in = self.act_ss2d(self.linear_ss2d(x_norm))
        ss2d_output = self.ss2d(ss2d_in, hw_shape)

        # 下路
        gate_output = self.act_gate(self.linear_gate(x_norm))

        # 3. 门控融合
        x_fused = ss2d_output * gate_output

        # 4. 后处理路径
        # 4.1 Linear after fusion
        post_fusion_linear_out = self.linear_post_fusion(x_fused)

        # 4.2 PAF
        # PAF 需要 2D 输入，所以需要 reshape
        paf_in_1 = post_fusion_linear_out.permute(0, 2, 1).reshape(B, C, H, W)
        # ss2d_output 也需要 reshape 和一个线性层来匹配维度
        # 根据SCSegamba的图，PAF的另一个输入是SS2D的输出，但维度是d_inner
        # 这里我们假设需要一个线性层将其投影回 embed_dims
        paf_in_2_proj = self.linear_post_fusion(ss2d_output) # 复用同一个线性层
        paf_in_2 = paf_in_2_proj.permute(0, 2, 1).reshape(B, C, H, W)
        paf_out = self.paf(paf_in_1, paf_in_2)

        # 4.3 GroupNorm
        # GroupNorm 需要 2D 输入
        gn_out = self.group_norm(paf_out)

        # 4.4 Final Linear
        # Reshape 回 token 序列
        post_paf_token = gn_out.permute(0, 2, 3, 1).reshape(B, L, C)
        post_processed_output = self.linear_post_paf(post_paf_token)

        # 5. 最终残差连接
        final_output = x_gbc + self.drop_path(post_processed_output)

        return final_output
