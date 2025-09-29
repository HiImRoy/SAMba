'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch
from torch import nn
from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFS
# 模型整体输入的维度为 [B, 3, 512, 512]   输出的维度为 [B, 1, 512, 512]
class Decoder(nn.Module):
    def __init__(self, backbone, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.MFS = MFS(8)

    def forward(self, samples):                     # samples: 一批图像，samples shape: [B, 3, 512, 512]
        outs_SAVSS = self.backbone(samples)         # 注意这里的outs_SAVSS是SAVSS的输出，是一个字典，包括了4个不同阶段的输出([B,16,512,512], [B,32,256,256], [B,64,128,128], [B,128,64,64])
        out = self.MFS(outs_SAVSS)                  # 调用 MFS 头部进行分割 out shape: [B,1,512,512]

        return out

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
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice

# Decoder 类封装了 backbone (SAVSS) 和 MFS 头部
def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    backbone = SAVSS(arch='Crack',               #arch='Crack' 使用 SAVSS.arch_zoo 中预定义的设置
                     out_indices=(0, 1, 2, 3),   # out_indices=(0, 1, 2, 3) 表示 SAVSS 将从主干网络的 4 个不同阶段/层输出特征。
                     drop_path_rate=0.2,
                     final_norm=True,
                     convert_syncbn=True)        # 实例化 SAVSS 作为 backbone
    model = Decoder(backbone, args)
    criterion = bce_dice(args) # BCEWithLogitsLoss(nn里的) 和 Dice Loss(文中说的) 的组合，然后由 args.BCELoss_ratio 和 args.DiceLoss_ratio 加权
    criterion.to(device)

    return model, criterion