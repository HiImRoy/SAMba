# 在这里写你的模型“组装图纸”
_base_ = [
    '../_base_/models/unet_r50-d16-aspp.py', # 这是一个占位符，后续需要修改
    '../_base_/datasets/crack_datasets.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
