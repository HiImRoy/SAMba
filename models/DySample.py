# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Original paper/code: https://github.com/tiny-smart/dysample

import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(module, mean=0, std=1, bias=0):
    """使用正态分布初始化模块的权重。"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    """使用常量初始化模块的权重。"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    """动态上采样模块。

    通过学习一个偏移场 (offset field) 来对特征图进行上采样，
    比传统的插值方法更具灵活性。
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale # 上采样倍数
        self.style = style # 采样风格 ('lp' 或 'pl')
        self.groups = groups # 分组数
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        # 用于学习偏移量的 1x1 卷积
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        # dyscope: 动态范围，用于调整偏移量的幅度
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        # 注册一个固定的初始位置 buffer
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        """初始化采样格点的相对位置。"""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        # `torch.meshgrid` 在新版本 PyTorch 中需要 `indexing` 参数
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        """核心采样函数，使用 F.grid_sample 执行采样。"""
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        # 创建基础坐标网格
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        # 归一化坐标到 [-1, 1] 范围
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        # 【最终修复】使用 pixel_shuffle 调整坐标以匹配上采样后的分辨率
        # 将 coords.view() 替换为 coords.reshape() 以处理因 transpose 导致的非连续张量
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        # 执行网格采样
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        """ 'lp' 风格的前向传播 (learn-pixel-shuffle) """
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        """ 'pl' 风格的前向传播 (pixel-shuffle-learn) """
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
''