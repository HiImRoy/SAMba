# Copyright (c) Roy. All rights reserved.

import torch.nn as nn
# [MODIFIED] 移除 mmcls 相关导入，以便在独立脚本中运行
# from mmcv.runner import BaseModule
# from ..builder import NECKS

# [FIXED] 使用从项目根目录开始的绝对导入路径，这是最稳健的方式
from mmcls.SAVSS_dev.models.samba_unet_modules.fusion import AdaptiveFusionModule


# @NECKS.register_module() # 暂时注释掉注册器
class AFNeck(nn.Module): # 暂时继承自 nn.Module
    """
    自适应融合颈部 (Adaptive Fusion Neck, AFNeck)。

    该模块作为一个颈部组件，接收来自两个不同主干网络 (Backbone)
    的多尺度特征金字塔，并在每个尺度上对它们进行自适应融合。
    """

    def __init__(self, in_channels_list, init_cfg=None):
        """
        初始化 AFNeck。

        Args:
            in_channels_list (list[int]): 一个包含各尺度输入通道数的列表。
                                          例如: [96, 192, 384, 768]
            init_cfg (dict, optional): 用于初始化的配置字典。默认为 None。
        """
        super(AFNeck, self).__init__()

        # 检查输入通道列表是否为包含4个元素的列表
        assert isinstance(in_channels_list, list) and len(in_channels_list) == 4, \
            f"in_channels_list 必须是一个包含4个整数的列表，但得到的是: {in_channels_list}"

        # 创建一个模块列表，其中包含4个并行的自适应融合模块
        self.fusion_modules = nn.ModuleList()
        for in_channels in in_channels_list:
            self.fusion_modules.append(
                AdaptiveFusionModule(in_channels=in_channels)
            )

    def forward(self, feats_sam, feats_mamba):
        """
        前向传播函数。

        Args:
            feats_sam (tuple[torch.Tensor]): 来自 SAM/Hiera 分支的特征金字塔。
            feats_mamba (tuple[torch.Tensor]): 来自 SAVSS/Mamba 分支的特征金字塔。

        Returns:
            list[torch.Tensor]: 一个包含4个融合后特征图的列表。
        """
        # 确保两个输入金字塔都包含4个尺度的特征
        assert len(feats_sam) == 4 and len(feats_mamba) == 4, \
            "输入的两个特征金字塔都必须包含4个尺度的特征"

        # 用于存储融合后特征的列表
        fused_feats = []

        # 遍历4个尺度，在每个尺度上进行融合
        for i in range(4):
            fused = self.fusion_modules[i](feats_sam[i], feats_mamba[i])
            fused_feats.append(fused)

        # 返回包含融合后特征的列表
        return fused_feats
