# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFSHead

# --- Focal Loss implementation ---
class FocalLoss(nn.Module):
    """
    A robust and numerically stable implementation of Focal Loss.
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        p_t = torch.exp(-bce_loss)
        modulating_factor = (1 - p_t) ** self.gamma
        alpha_weight_factor = (y_true * self.alpha + (1 - y_true) * (1 - self.alpha))
        focal_loss = alpha_weight_factor * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
        # --- [FIXED] Hard-coded the Focal Loss parameters to remove dependency on args ---
        self.bce_fn = FocalLoss(gamma=2.0, alpha=0.25)
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = F.interpolate(y_pred, size=y_true.shape[-2:], mode='bilinear', align_corners=False)

        y_true = (y_true > 0).float()

        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice

# The build and Decoder classes are not part of the main SAMbaCrack model flow
# but are kept for potential other uses.
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
