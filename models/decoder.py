# Author: Roy
# Copyright (c) Roy. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer


class SAVSSDecoderBlock(nn.Module):
    """
    功能: 一个解码器块，包含多个SAVSS层用于特征处理。
    """
    def __init__(self, dim, depth=2, savss_layer_args={}):
        super().__init__()
        # [关键] 确保 savss_layer_args 被正确传递给 SAVSS_Layer
        self.layers = nn.ModuleList([
            SAVSS_Layer(embed_dims=dim, **savss_layer_args) for _ in range(depth)
        ])

    def forward(self, x, H, W):
        for layer in self.layers:
            x = layer(x, hw_shape=(H, W))
        return x


class SAVSSUp(nn.Module):
    """
    功能: 一个完整的上采样+融合+处理模块。
    """
    def __init__(self, in_dim, out_dim, depth=2, savss_layer_args={}):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.proj = nn.Linear(out_dim * 2, out_dim)
        # [关键] 将接收到的 savss_layer_args 传递给 SAVSSDecoderBlock
        self.block = SAVSSDecoderBlock(dim=out_dim, depth=depth, savss_layer_args=savss_layer_args)

    def forward(self, x, skip, H, W):
        x = x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2)
        x_up = self.upsample(x)
        x_up_token = x_up.permute(0, 2, 3, 1).flatten(1, 2)
        skip_token = skip.flatten(2, 3).transpose(1, 2)
        fused = torch.cat([x_up_token, skip_token], dim=2)
        fused = self.proj(fused)
        new_H, new_W = H * 2, W * 2
        output = self.block(fused, new_H, new_W)
        return output, new_H, new_W


class SAVSS_UNet_Decoder(nn.Module):
    """
    功能: 完整的、对称的SAVSS U-Net解码器。
    """
    def __init__(self, decoder_dims=[512, 256, 128, 64], decoder_depths=[2, 2, 2, 2], savss_layer_args={}):
        super().__init__()
        self.decoder_dims = decoder_dims

        # --- 创建3个上采样阶段 ---
        self.up_stages = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.up_stages.append(
                SAVSSUp(
                    in_dim=decoder_dims[i],
                    out_dim=decoder_dims[i + 1],
                    depth=decoder_depths[i],
                    # [关键] 将整个配置字典传递给SAVSSUp模块
                    savss_layer_args=savss_layer_args
                )
            )

    # ... forward 和 segmentation_head 部分保持不变 ...
        self.segmentation_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dims[-1] // 2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, features):
        x = features[0]
        B, C, H, W = x.shape
        x_token = x.flatten(2, 3).transpose(1, 2)
        x_token, H, W = self.up_stages[0](x_token, features[1], H, W)
        x_token, H, W = self.up_stages[1](x_token, features[2], H, W)
        x_token, H, W = self.up_stages[2](x_token, features[3], H, W)
        final_features = x_token.transpose(-2, -1).reshape(B, self.decoder_dims[-1], H, W)
        logits = self.segmentation_head(final_features)
        return logits

# ... (DiceLoss 和 bce_dice 保持不变) ...
class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    def forward(self, x, y):
        tp = (x * y).sum(self.dims); fp = (x * (1 - y)).sum(self.dims); fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth); dc = dc.mean()
        return 1 - dc
class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss(); self.dice_fn = DiceLoss(); self.args = args
    def forward(self, y_pred, y_true):
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = F.interpolate(y_pred, size=y_true.shape[-2:], mode='bilinear', align_corners=False)
        bce = self.bce_fn(y_pred, y_true); dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice