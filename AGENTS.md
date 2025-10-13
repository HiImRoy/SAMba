项目简报: SAMbaCrack-AF 模型架构与数据流详解
你好 Gemini，这是 SAMbaCrack-AF 模型的最终设计蓝图。请你仔细阅读并完全理解这个架构。后续我们所有的讨论和编码都将以此文档为准。
第一部分: 编码器 (Backbone) - SAMbaCrackEncoder
职责: 模型的特征提取引擎，由两个完全独立、并行的分支组成。
1.1 SAM/Hiera 分支 (冻结的语义编码器)
角色: 利用强大的预训练权重，提供具有丰富高级语义信息的特征。其参数在训练中被冻结。
输入: 原始图像 (B, 3, 448, 448)。
结构: 层级式视觉 Transformer (Hiera)。
输出 (feats_sam): 一个包含4个多尺度特征图（经过 Refiner 和 Adapter 处理）的元组：
[0]: (B, 96, 112, 112)
[1]: (B, 192, 56, 56)
[2]: (B, 384, 28, 28)
[3]: (B, 768, 14, 14)
1.2 SAVSS/Mamba 分支 (可训练的细节编码器)
角色: 从头学习任务特定的特征，特别是裂缝的纹理、走向等细节信息。
结构: 这是一个特殊的“编码器-解码器”混合体。
数据流:
A. 初始编码 (非分层):
PatchEmbed (stride=8): (B, 3, 448, 448) -> (B, 3136, 256)。
4个串联的 SAVSS_Layer 在固定的 (B, 3136, 256) 序列上进行深度特征提取。
B. 并行解码 (生成金字塔):
每一个 SAVSS_Layer 的输出都会被送入一个专属的解码头 savss_decoder_heads。
以第2个解码头为例:
输入: 第2个 SAVSS_Layer 的输出 (B, 3136, 256)。
Reshape & Permute: (B, 3136, 256) -> (B, 256, 56, 56)。
降通道 (1x1 Conv): (B, 256, 56, 56) -> (B, 192, 56, 56)。
调尺寸 (Upsample): (B, 192, 56, 56) -> (B, 192, 56, 56) (此阶段不变)。
输出: 得到 feats_mamba[1]。
输出 (feats_mamba): 一个包含4个特征图的元组，其维度与 feats_sam 完全匹配。
第二部分: 颈部 (Neck) - AFNeck
职责: 模型的“脊柱”，负责将来自两个主干网络的特征进行智能、自适应的融合。
输入: feats_sam 和 feats_mamba 两个并行的特征金字塔。
结构: 包含4个并行的 AdaptiveFusionModule。
数据流 (以 Stage 1 为例):
fusion_modules[0] 接收 feats_sam[0] 和 feats_mamba[0] (均为 (B, 96, 112, 112))。
内部执行 Concat -> 1x1 Conv -> 3x3 Conv -> Sigmoid 生成门控 gate。
最终通过 Gated Fusion (sam * gate + mamba * (1 - gate)) 进行融合。
输出 (fused_feats): 一个单一的、融合后的特征金字塔，维度与输入金字塔完全相同。
第三部分: 解码器头 (Head) - MFSHead (【全分辨率版】)
职责: 将融合后的多尺度特征直接解码为最终的全分辨率像素级预测。
输入: fused_feats ((c1, c2, c3, c4))，其中 c1 是 (B, 96, 112, 112)。
【核心修改】内部数据流与维度变化 (假设 embedding_dim 由我指定):
A. 并行通道对齐 (MLPs):
4个独立的 MLP (1x1 Conv) 将 [96, 192, 384, 768] 的通道数统一映射到 embedding_dim。
示例输出: c1_p -> (B, embedding_dim, 112, 112) 等。
B. 并行空间对齐 (上采样至全分辨率):
使用 DySample (或 F.interpolate) 将4个处理后的特征图全部上采样到 448x448。
这个embedding_dim为8
upsample_c1(c1_p) -> (B, embedding_dim, 448, 448) (上采样x4)
upsample_c2(c2_p) -> (B, embedding_dim, 448, 448) (上采样x8)
upsample_c3(c3_p) -> (B, embedding_dim, 448, 448) (上采样x16)
upsample_c4(c4_p) -> (B, embedding_dim, 448, 448) (上采样x32)
C. 全分辨率拼接 (Concat):
将上述4个 (B, embedding_dim, 448, 448) 的特征图拼接 -> (B, embedding_dim * 4, 448, 448)。
D. 全分辨率深度融合:
fusion_gbc: (B, embedding_dim * 4, 448, 448) -> (B, embedding_dim * 4, 448, 448)。
fusion_conv (1x1 Conv): (B, embedding_dim * 4, 448, 448) -> (B, embedding_dim, 448, 448)。
E. 全分辨率预测:
final_conv (1x1 Conv): (B, embedding_dim, 448, 448) -> (B, 1, 448, 448)。
输出 (logits): 直接输出最终的全分辨率预测图 (B, 1, 448, 448)。
第四部分: 最终输出与损失计算
最终输出: MFSHead 的输出 logits 不再需要额外的上采样步骤。
损失计算: logits ((B, 1, 448, 448)) 直接与 Ground Truth ((B, 1, 448, 448)) 一起送入损失函数。
确认: 请确认你已理解这份更新后的蓝图，特别是 MFSHead 内部的“全分辨率解码”逻辑。
我的第一个任务是: [在这里写下你的第一个具体任务，例如：“请为我重构 MFSHead 的代码，以实现上述全分辨率解码功能。”]