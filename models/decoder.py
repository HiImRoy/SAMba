# Copyright (c) Roy. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFSHead
import numpy as np

# --- Focal Loss implementation (Kept for potential other uses) ---
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

# ==============================================================================
# TASK 1: Boundary Weight Map Generation Function
# ==============================================================================

def generate_boundary_map(mask: torch.Tensor, boundary_weight: float = 3.0) -> torch.Tensor:
    """
    Generates a boundary weight map from a ground truth segmentation mask.
    Uses a Sobel operator to detect edges and assigns a higher weight to them.
    """
    mask = mask.float()
    device = mask.device

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)

    grad = torch.sqrt(grad_x**2 + grad_y**2)
    weight_map = torch.ones_like(mask)
    is_boundary = (grad > 0)
    weight_map[is_boundary] = boundary_weight

    return weight_map

# ==============================================================================
# TASK 2: Boundary Refinement Loss (BRL) Class
# ==============================================================================

class BoundaryRefinementLoss(nn.Module):
    """
    Boundary Refinement Loss (BRL).
    Applies a weight to the standard BCE Loss, focusing the model on the accuracy of target boundaries.
    """
    def __init__(self, boundary_weight: float = 3.0):
        super(BoundaryRefinementLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight_map = generate_boundary_map(targets, self.boundary_weight)
        raw_bce_loss = self.bce_loss(logits, targets)
        weighted_bce_loss = raw_bce_loss * weight_map
        final_loss = weighted_bce_loss.mean()
        return final_loss

# ==============================================================================
# TASK 3: Reconstructed Composite Loss Function
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss, commonly used for semantic segmentation to measure sample similarity.
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

class CompositeLoss(nn.Module):
    """
    A composite loss function that combines BCE Loss, Dice Loss, and Boundary Refinement Loss.
    """
    def __init__(self, args):
        super(CompositeLoss, self).__init__()
        self.bce_weight = args.bce_weight
        self.dice_weight = args.dice_weight
        self.brl_weight = args.brl_weight

        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = DiceLoss()
        self.brl_loss_fn = BoundaryRefinementLoss() # Uses default boundary_weight=3.0

    def forward(self, y_pred, y_true):
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = F.interpolate(y_pred, size=y_true.shape[-2:], mode='bilinear', align_corners=False)

        y_true = (y_true > 0).float()

        loss_bce = self.bce_loss_fn(y_pred, y_true)
        loss_dice = self.dice_loss_fn(y_pred, y_true)
        loss_brl = self.brl_loss_fn(y_pred, y_true)

        total_loss = (self.bce_weight * loss_bce +
                      self.dice_weight * loss_dice +
                      self.brl_weight * loss_brl)
        return total_loss


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
    # MODIFIED: Instantiate the new CompositeLoss
    criterion = CompositeLoss(args)
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
