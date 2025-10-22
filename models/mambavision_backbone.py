# models/mamba_vision_backbone.py
# 备注: 该文件由 Gemini 重构，作为一个纯粹的分割任务主干网络。
# [BACKBONE_CHANGE] 移除了所有与 timm 模型注册、预训练配置 (_cfg, default_cfgs) 和复杂的检查点加载相关的辅助函数。
# [BACKBONE_CHANGE] 目标是创建一个独立的、更轻量的、专注于特征提取的文件。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# --- 辅助模块和函数 ---

class Mlp(nn.Module):
    """
    标准的多层感知机（MLP）模块。
    为了使该文件自包含，代码从 TIMM 库复制而来。
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        Args:
            in_features (int): 输入特征维度。
            hidden_features (int): 隐藏层特征维度。默认为输入特征维度。
            out_features (int): 输出特征维度。默认为输入特征维度。
            act_layer (nn.Module): 激活函数层。
            drop (float): Dropout 的概率。
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状 (B, N, C)。
        Returns:
            Tensor: 输出张量，形状 (B, N, C)。
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将特征图 (feature map) 分割成不重叠的窗口。
    这是 Vision Transformer 中常用的操作，用于在局部窗口内计算自注意力。

    Args:
        x (Tensor): 输入特征图，形状为 (B, C, H, W)。
        window_size (int): 窗口的大小。

    Returns:
        Tensor: 分割后的窗口，形状为 (num_windows*B, window_size*window_size, C)。
    """
    B, C, H, W = x.shape
    # 将 H 和 W 维度拆分为 网格数 x 窗口大小
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    # 重新排列维度，使窗口内的 patch 连续存储
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将分割后的窗口重新组合成原始的特征图。
    这是 window_partition 的逆操作。

    Args:
        windows (Tensor): 分割后的窗口，形状为 (num_windows*B, window_size*window_size, C)。
        window_size (int): 窗口的大小。
        H (int): 原始特征图的高度。
        W (int): 原始特征图的宽度。

    Returns:
        Tensor: 恢复后的特征图，形状为 (B, C, H, W)。
    """
    # 计算原始的批次大小 B
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 将窗口数据 reshape 回多维形式
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 重新排列维度，恢复 (B, C, H, W) 的格式
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


# --- 核心网络层定义 ---

class Downsample(nn.Module):
    """
    带步长（stride）的卷积下采样模块。
    通过一个 3x3 卷积（步长为2）将特征图的空间分辨率减半，同时将通道数翻倍。
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): 输入通道数。
        """
        super().__init__()
        # [BACKBONE_CHANGE] 原始实现中，此模块有一个 keep_dim 参数，这里将其简化，
        # 因为在骨干网络中，下采样总是伴随着通道数的翻倍。
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        # 输入 x: (B, C, H, W)
        # 输出: (B, 2*C, H/2, W/2)
        return self.reduction(x)


class PatchEmbed(nn.Module):
    """
    图像到 Patch 的嵌入模块。
    将输入的图像通过两个带步长的卷积层，将其转换为一系列 Patch 嵌入（token）。
    总下采样率为 4。
    """
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans (int): 输入图像的通道数 (例如，RGB为3)。
            in_dim (int): 第一个卷积层输出的通道数。
            dim (int): 第二个卷积层输出的最终通道数。
        """
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False), # H/2, W/2
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False), # H/4, W/4
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Input: x, 形状 (B, C_in, H, W)
        Output: 形状 (B, dim, H/4, W/4)
        """
        return self.conv_down(x)


class ConvBlock(nn.Module):
    """
    残差卷积块。
    一个简单的包含两个卷积层、归一化和激活函数的残差块。
    """
    def __init__(self, dim, drop_path=0., layer_scale=None):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)

        # LayerScale: 一个可学习的参数，用于缩放残差连接的输出
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale is not None else 1.0
        # DropPath: 随机深度，在训练时随机“丢弃”整个残差块
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # 应用 LayerScale
        if isinstance(self.gamma, torch.Tensor):
            x = x * self.gamma.view(1, -1, 1, 1)
        else:
            x = x * self.gamma
        # 添加残差连接和 DropPath
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    """
    MambaVision 核心混合器。
    这是实现状态空间模型（SSM）的核心部分，用于替代传统 Transformer 中的自注意力机制。
    它通过一个高效的选择性扫描操作来捕捉长距离依赖。
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", **kwargs):
        """
        [BACKBONE_CHANGE] 简化了 __init__ 的参数列表。原始实现包含许多用于微调 SSM 行为的参数
        (如 dt_min, dt_max, dt_init, dt_scale, bias, use_fast_path 等)。
        这里保留了核心参数，使模块更易于理解和使用。

        Args:
            d_model (int): 模型的主维度。
            d_state (int): 状态维度 N。
            d_conv (int): 1D 卷积核的大小。
            expand (int): 扩展因子 E，用于计算内部维度 d_inner。
            dt_rank (int or "auto"): 时间步参数 Delta (Δ) 的秩。
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model) # 内部维度 E*D
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影层，将 d_model 扩展到 d_inner
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        # 用于生成 SSM 参数 (Δ, B, C) 的投影层
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        # 用于从 dt_rank 生成 Δ 的投影层
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        # 初始化 Δ (dt) 的偏置
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner // 2) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # 初始化状态矩阵 A 和参数 D
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner // 2)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))

        # [BACKBONE_CHANGE] 原始实现使用 F.conv1d，这里为了结构清晰，明确定义了 nn.Conv1d 模块。
        self.conv1d_x = nn.Conv1d(self.d_inner // 2, self.d_inner // 2, kernel_size=d_conv, padding='same', groups=self.d_inner // 2, bias=True)
        self.conv1d_z = nn.Conv1d(self.d_inner // 2, self.d_inner // 2, kernel_size=d_conv, padding='same', groups=self.d_inner // 2, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (Tensor): 输入张量，形状 (B, L, D)，其中 L 是序列长度，D 是 d_model。
        """
        _, seqlen, _ = hidden_states.shape
        # 1. 输入投影和分割
        xz = self.in_proj(hidden_states) # (B, L, D) -> (B, L, D_inner)
        xz = rearrange(xz, "b l d -> b d l") # (B, L, D_inner) -> (B, D_inner, L)
        x, z = xz.chunk(2, dim=1) # 分割为 x 和 z 两部分, 形状均为 (B, D_inner/2, L)
        A = -torch.exp(self.A_log.float())

        # 2. 1D 卷积
        # [BACKBONE_CHANGE] 调用定义的 nn.Conv1d 模块，而不是 F.conv1d。
        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))

        # 3. 计算 SSM 参数 (Δ, B, C)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d")) # (B, D_inner/2, L) -> (B*L, D_inner/2)
        dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen) # (B*L, dt_rank) -> (B, L, D_inner/2) -> (B, D_inner/2, L)
        B_param = rearrange(B_param, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_param = rearrange(C_param, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # 4. 执行选择性扫描 (Selective Scan)
        y = selective_scan_fn(x, dt, A, B_param, C_param, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

        # 5. Gating 和输出投影
        y = torch.cat([y, z], dim=1) # 拼接 y 和 z
        y = rearrange(y, "b d l -> b l d") # (B, D_in, L) -> (B, L, D_in)
        return self.out_proj(y) # (B, L, D_in) -> (B, L, D)


class Attention(nn.Module):
    """
    标准的多头自注意力（Multi-Head Self-Attention）模块。
    用于 Transformer 块中。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        # [BACKBONE_CHANGE] 简化了 __init__ 参数，移除了 qk_norm 和 norm_layer，因为在这个实现中未使用。
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5 # 缩放因子

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # 1. 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # [BACKBONE_CHANGE] 原始实现中有一个 fused_attn 标志，并依赖 PyTorch 2.0 的 F.scaled_dot_product_attention。
        # 这里为了兼容性（PyTorch < 2.0）和简化，直接使用手动实现的 attention。
        # 2. 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 3. 加权求和
        x = (attn @ v)

        # 4. 拼接多头并进行输出投影
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    混合块，可以是 Mamba 块或 Transformer 块。
    这是一个标准的 Pre-Norm 结构: Norm -> Mixer -> DropPath -> Residual -> Norm -> MLP -> DropPath -> Residual
    """
    def __init__(self, dim, num_heads, is_transformer_block=False, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., layer_scale=None):
        # [BACKBONE_CHANGE] 极大地简化了 __init__ 参数。原始实现依赖一个 `counter` 和 `transformer_blocks` 列表来决定
        # mixer 类型。这里改为使用一个更直观的布尔标志 `is_transformer_block`。
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 根据 is_transformer_block 标志选择使用 Attention 还是 MambaVisionMixer
        if is_transformer_block:
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.mixer = MambaVisionMixer(d_model=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale is not None else 1.0
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale is not None else 1.0

    def forward(self, x):
        # 第一个残差连接
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        # 第二个残差连接
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# [BACKBONE_CHANGE] 将原始的 `MambaVisionLayer` 重命名为 `MambaVisionStage`，使其语义更清晰。
class MambaVisionStage(nn.Module):
    """
    MambaVision 的一个阶段（Stage）。
    一个 Stage 包含多个块（ConvBlock 或 Block），并在最后可选地进行下采样。
    """
    def __init__(self, dim, depth, num_heads, window_size, is_conv_stage=False, downsample=True, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., layer_scale=None, transformer_block_indices=[]):
        # [BACKBONE_CHANGE] 简化了 __init__ 参数，并改变了块类型的决定方式。
        # 原始实现有一个 `conv` 标志，这里改为 `is_conv_stage`，逻辑更清晰。
        super().__init__()

        # 根据 is_conv_stage 标志选择使用纯卷积块还是混合块
        if is_conv_stage:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim, drop_path=drop_path[i], layer_scale=layer_scale) for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    is_transformer_block=(i in transformer_block_indices), # 决定当前块是 Mamba 还是 Attention
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale
                ) for i in range(depth)
            ])

        self.downsample = Downsample(dim=dim) if downsample else None
        self.window_size = window_size
        self.is_conv_stage = is_conv_stage

    def forward(self, x):
        B, C, H, W = x.shape

        # Store original H, W for window_reverse if needed
        original_H, original_W = H, W

        # 如果不是纯卷积阶段，需要进行窗口划分
        if not self.is_conv_stage:
            # 对特征图进行填充，使其可以被 window_size 整除
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
            # (B, C, H, W) -> (B*num_windows, L, C)
            x = window_partition(x, self.window_size)

        # 依次通过该阶段的所有块
        for blk in self.blocks:
            x = blk(x)

        # 如果不是纯卷积阶段，需要将窗口恢复
        if not self.is_conv_stage:
            # (B*num_windows, L, C) -> (B, C, Hp, Wp)
            x = window_reverse(x, self.window_size, Hp, Wp)
            # 移除之前填充的部分
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :original_H, :original_W].contiguous()

        # This 'x_pre_downsample' is the feature the user wants for FCM
        x_pre_downsample = x

        # 如果需要，进行下采样
        x_post_downsample = x # Default to pre-downsample if no downsample layer
        if self.downsample:
            x_post_downsample = self.downsample(x)

        return x_pre_downsample, x_post_downsample


# --- 主模型 MambaVision ---
class MambaVision(nn.Module):
    """
    MambaVision 主干网络 (为特征提取而重构)。
    该模型由一个 Patch 嵌入层和多个 MambaVisionStage 组成。
    """
    def __init__(self, dim, in_dim, depths, window_size, num_heads, mlp_ratio=4., drop_path_rate=0.2, in_chans=3, qkv_bias=True, layer_scale=None, **kwargs):
        # [BACKBONE_CHANGE] 简化了 __init__ 参数，移除了分类头相关的 `num_classes` 等参数。
        super().__init__()
        # 1. Patch 嵌入层
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        # 2. 构建多个 Stage
        # 计算每个块的 DropPath 概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # [BACKBONE_CHANGE] 原始实现中这一部分叫做 `self.levels`，这里改为 `self.stages`，语义更清晰。
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            # 确定哪些块是 Transformer 块 (论文策略：每个阶段的后半部分)
            num_transformer_blocks = depths[i] // 2
            transformer_indices = list(range(depths[i] - num_transformer_blocks, depths[i]))

            stage = MambaVisionStage(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                is_conv_stage=(i < 2),  # 前两个阶段是纯卷积
                # [BACKBONE_CHANGE] 这是关键修改，确保所有阶段都下采样，以生成适用于U-Net解码器的新输入。
                downsample=True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                layer_scale=layer_scale if i >= 2 else None,  # 仅对 Mamba/Transformer 阶段使用 LayerScale
                transformer_block_indices=transformer_indices
            )
            self.stages.append(stage)

        # 3. 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ 初始化模型权重 """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        [BACKBONE_CHANGE] 这是最核心的修改。原始模型的 forward 方法会经过分类头输出一个分类结果。
        作为骨干网络，我们修改 forward 方法，使其返回一个包含每个阶段输出特征图的元组 (tuple)。
        这为后续的解码器（如 U-Net）提供了多尺度的特征信息。

        前向传播函数。
        
        Args:
            x (Tensor): 输入图像，形状 (B, 3, H, W)。
            
        Returns:
            tuple[Tensor]: 一个包含每个 stage *下采样前* 输出特征图的元组。
        """
        x = self.patch_embed(x)

        outs = []
        for i, stage in enumerate(self.stages):
            # 每个阶段现在返回 (下采样前的特征, 下采样后的特征)
            features_before_downsample, features_after_downsample = stage(x)
            outs.append(features_before_downsample) # 收集下采样前的特征
            x = features_after_downsample # 下一个阶段的输入是当前阶段下采样后的特征

        return tuple(outs)


# [BACKBONE_CHANGE] 用一个统一的、更简洁的工厂函数 `create_mamba_vision_backbone` 
# 替换了原始文件中为每个模型变体（如 mamba_vision_T, mamba_vision_S 等）编写的大量重复的工厂函数。
# 这个函数还包含了简化的预训练权重加载逻辑。
def create_mamba_vision_backbone(variant='B', pretrained=False, pretrained_path='', **kwargs):
    """
    创建并选择性地加载 MambaVision 主干网络的预训练权重。

    Args:
        variant (str): 模型版本, 可选 'T', 'S', 'B', 'L'。
        pretrained (bool): 是否加载预训练权重。
        pretrained_path (str): 预训练权重文件的路径。
        **kwargs: 要覆盖的额外模型参数 (例如, drop_path_rate)。'depths' 会被忽略。

    Returns:
        nn.Module: MambaVision 模型实例。
    """
    # 不同模型变体的配置
    model_configs = {
        'T': {'depths': [1, 3, 8, 4], 'num_heads': [2, 4, 8, 16], 'dim': 80, 'in_dim': 32},
        'S': {'depths': [3, 3, 7, 5], 'num_heads': [2, 4, 8, 16], 'dim': 96, 'in_dim': 64},
        'B': {'depths': [3, 3, 10, 5], 'num_heads': [2, 4, 8, 16], 'dim': 128, 'in_dim': 64, 'layer_scale': 1e-5},
        'L': {'depths': [3, 3, 10, 5], 'num_heads': [4, 8, 16, 32], 'dim': 196, 'in_dim': 64, 'layer_scale': 1e-5},
    }

    config = model_configs.get(variant.upper())
    if config is None:
        raise ValueError(f"无效的变体 '{variant}'. 请从 {list(model_configs.keys())} 中选择。")

    # 通用参数
    config['window_size'] = [8, 8, 14, 7]

    # [重构] 强制使用变体自带的深度，忽略外部传入的 'depths'
    if 'depths' in kwargs:
        print(f"信息: 强制根据变体 '{variant}' 设置阶段深度。使用深度: {config['depths']}。"
              f"外部传入的 'depths' 参数将被忽略。")
        del kwargs['depths']

    # 使用 kwargs 覆盖其他默认配置 (如 drop_path_rate)
    config.update(kwargs)

    # 创建模型实例
    model = MambaVision(**config)

    # 加载预训练权重
    if pretrained:
        if not pretrained_path:
            raise ValueError("当 `pretrained=True` 时，必须提供预训练权重的路径。")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)  # 兼容不同格式的权重文件

            # [BACKBONE_CHANGE] 加载权重时，移除与分类任务相关的最终层（如 head, norm, avgpool）的权重。
            for key in list(state_dict.keys()):
                if key.startswith('head.') or key.startswith('norm.') or key.startswith('avgpool.'):
                    del state_dict[key]

            # 加载权重 (strict=False 允许忽略不匹配的键)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"已从 '{pretrained_path}' 加载 MambaVision-{variant} 的预训练权重。")
            if missing_keys:
                print("缺失的键:", missing_keys)
            if unexpected_keys:
                print("意外的键:", unexpected_keys)

        except Exception as e:
            print(f"加载预训练权重时出错: {e}")

    return model
