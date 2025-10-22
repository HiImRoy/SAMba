# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# REFACTOR: This file was created by refactoring from the old decoder.py

import torch
from torch import nn
import torch.nn.functional as F

# --- Loss Functions ---

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        # 如果预测和标签的尺寸不匹配，将预测上采样到标签的尺寸
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = F.interpolate(y_pred, size=y_true.shape[-2:], mode='bilinear', align_corners=False)

        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice
