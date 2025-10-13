# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFSHead

class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mfs_head = MFSHead(in_channels=[16, 32, 64, 128])

    def forward(self, samples):
        outs_SAVSS = self.backbone(samples)
        out = self.mfs_head(outs_SAVSS)
        _, _, H, W = samples.shape
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        intersection = (x * y).sum(self.dims)
        cardinality = x.sum(self.dims) + y.sum(self.dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = F.interpolate(y_pred, size=y_true.shape[-2:], mode='bilinear', align_corners=False)

        # --- [FIX] Ensure target mask is binary (0 or 1) --- #
        # This handles masks loaded with pixel values like [0, 255] from bmp/jpg/png files.
        # It converts any non-zero value in the mask to 1.0, ensuring a correct Dice Loss calculation.
        y_true = (y_true > 0).float()

        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice

def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    backbone = SAVSS(arch='Crack',
                     out_indices=(0, 1, 2, 3),
                     drop_path_rate=0.2,
                     final_norm=True,
                     convert_syncbn=True)
    model = Decoder(backbone, args)
    criterion = bce_dice(args)
    criterion.to(device)

    return model, criterion
