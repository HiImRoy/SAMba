# Copyright (c) Roy. All rights reserved.

# --- 模型基础配置 (可根据需要修改) ---
_base_ = [
    '../_base_/datasets/ade20k.py', # 示例数据集配置
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py' # 示例训练策略配置
]

# --- 自定义模型架构：SAMbaCrack-AF ---
# 该配置将我们自定义的 Backbone, Neck, Head 组装在一起
model = dict(
    # 1. 模型类型定义
    # 使用我们自定义的 SAMbaCrackAF 模型，它知道如何处理双路主干的输出
    type='SAMbaCrackAF', 
    
    # 2. 编码器 (Backbone)
    # 使用我们新创建的、独立的双分支主干网络
    backbone=dict(
        type='SAMbaCrackEncoder',
        # --- 编码器初始化参数 ---
        sam_dims=[96, 192, 384, 768],
        token_dim=256,
        patch_size=8,
        load_height=448,
        load_width=448,
        # 可以根据需要暴露更多参数到配置中，例如：
        # hiera_depths=(2, 2, 6, 2),
        # savss_drop_path_rate=0.1
    ),
    
    # 3. 颈部 (Neck)
    # 使用 AFNeck 融合来自双分支编码器的特征
    neck=dict(
        type='AFNeck',
        # in_channels_list 必须与 backbone 输出的每个尺度的通道数完全匹配
        in_channels_list=[96, 192, 384, 768]
    ),
    
    # 4. 解码器头 (Decode Head)
    # 使用 MFSHead 将融合后的特征解码为分割图
    decode_head=dict(
        type='MFSHead',
        # in_channels_list 必须与 neck 输出的每个尺度的通道数完全匹配
        in_channels_list=[96, 192, 384, 768],
        # embedding_dim 是 MFSHead 内部统一处理的维度
        embedding_dim=256,
        # 分割任务的类别数。对于二分类（如裂缝/背景），通常设置为2。
        num_classes=2, 
        # 对齐角点，MMSeg 推荐的解码器设置
        align_corners=False,
        # 定义损失函数
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, # 使用 Softmax
            loss_weight=1.0
        )
    ),
    
    # --- 训练和测试设置 ---
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# --- 数据集和评估配置 (示例) ---
# data = dict(samples_per_gpu=2, workers_per_gpu=2)
# evaluation = dict(metric='mIoU')
