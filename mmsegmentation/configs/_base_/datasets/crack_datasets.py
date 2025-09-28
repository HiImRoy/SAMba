"""# 1. 数据集通用设置
dataset_type = 'BaseSegDataset'
# 注意：data_root 在下面的具体数据集中单独定义
crop_size = (512, 512)
# 类别和调色板
metainfo = dict(classes=('background', 'crack'), palette=[[0, 0, 0], [255, 0, 0]])

# 2. 训练数据处理流程 (Pipeline)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 3. 测试数据处理流程 (Pipeline)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # 在Resize之后加载标注，因为GT不需要Resize
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 4. 定义两个数据集的训练配置
crack500_train_dataset = dict(
    type=dataset_type,
    data_root='data/crack500',
    data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=train_pipeline,
    metainfo=metainfo
)

deepcrack_train_dataset = dict(
    type=dataset_type,
    data_root='data/deepcrack',
    data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
    pipeline=train_pipeline,
    metainfo=metainfo
)

# 5. 数据加载器 (Dataloader)
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # 使用 ConcatDataset 合并两个训练集
    dataset=dict(
        type='ConcatDataset',
        datasets=[crack500_train_dataset, deepcrack_train_dataset]
    )
)

# 验证和测试暂时只使用 Crack500 数据集
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='data/crack500',
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
        metainfo=metainfo
    )
)
test_dataloader = val_dataloader

# 6. 评估指标 (Evaluator)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
""