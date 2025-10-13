# Copyright (c) Roy. All rights reserved.

from .encoder_decoder import EncoderDecoder
from ..builder import SEGMENTORS


@SEGMENTORS.register_module()
class SAMbaCrackAF(EncoderDecoder):
    """
    SAMbaCrack-AF 模型，一个为双分支主干网络设计的编码器-解码器。

    该模型继承自标准的 EncoderDecoder，但重写了 `extract_feat` 方法，
    以正确处理其特殊的双分支主干网络 (SAMbaCrackEncoder) 返回的两个
    并行的特征金字塔。
    """

    def __init__(self, **kwargs):
        super(SAMbaCrackAF, self).__init__(**kwargs)

    def extract_feat(self, img):
        """
        重写特征提取函数，这是连接 Backbone -> Neck 的关键。

        标准的 `extract_feat` 只期望主干网络返回一个输出，而我们的主干网络
        返回两个。此方法将正确地接收这两个输出，并将它们传递给颈部模块
        进行融合，然后返回一个融合后的特征金字塔，以供后续的解码头使用。

        Args:
            img (torch.Tensor): 输入的图像张量。

        Returns:
            list[torch.Tensor]: 经过颈部模块融合后的单路特征金字塔。
        """
        # 1. 从主干网络 (self.backbone) 获取两个并行的特征金字塔
        # feats_sam 和 feats_mamba 的维度应该完全匹配
        feats_sam, feats_mamba = self.backbone(img)

        # 2. 将两个金字塔送入颈部 (self.neck) 进行融合
        # self.neck (AFNeck) 会在内部对两个金字塔的对应层进行处理
        fused_feats = self.neck(feats_sam, feats_mamba)

        # 3. 返回融合后的、单一的特征金字塔
        # 这个返回值将无缝地对接父类中后续的解码头 (decode_head) 处理流程
        return fused_feats
