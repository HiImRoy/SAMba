'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org

*** Modified for correct binary mask output ***
'''

import numpy as np
import torch
import argparse
import os
import cv2
from tqdm import tqdm
from datasets import create_dataset
from models import build_model
from main import get_args_parser

# 1. 创建参数解析器
parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()

# --- START: 可修改区域 (根据您的设置) ---

# 2. 明确设置运行阶段和数据集路径
args.phase = 'test'
#    确保这里的路径是相对于您项目根目录的正确路径
args.dataset_path = './data/CRACK500'

# 3. 明确设置加载的图像尺寸以匹配您的数据
args.load_width = 448
args.load_height = 448

# --- END: 可修改区域 ---

if __name__ == '__main__':
    args.batch_size = 1
    device = torch.device(args.device)

    # 4. 创建数据集 (现在会使用上面更新后的args)
    print("正在创建测试数据集...")
    test_dl = create_dataset(args)
    print(f"数据集创建成功，共找到 {len(test_dl)} 张测试图片。")

    model, _ = build_model(args) # criterion在测试时不需要，可以用_忽略

    # --- START: 修改权重加载逻辑 ---
    # 使用 --resume 参数来指定权重文件路径
    # 如果没有通过命令行指定 --resume，则使用一个默认路径
    if not args.resume:
        # 默认的权重文件路径，用户可以根据需要修改
        default_checkpoint_path = "results/samba_v19.0/20251020-190044_SAMbaCrack_DeepCrack miou 0.92/weights/checkpoint_best.pth"
        print(f"未通过 --resume 参数指定权重文件，将尝试加载默认路径: {default_checkpoint_path}")
        args.resume = default_checkpoint_path

    # 检查权重文件是否存在
    if not os.path.exists(args.resume):
        print(f"错误：权重文件未找到！请检查路径: {args.resume}")
        exit()

    print(f"正在从 {args.resume} 加载权重...")
    # 添加 map_location 以确保设备兼容性
    state_dict = torch.load(args.resume, map_location=device)
    
    # 检查 state_dict 中是否包含 'model' 键，这通常是训练 checkpoint 的格式
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        # 如果没有 'model' 键，则尝试直接加载整个 state_dict
        # 这可能是直接保存的模型 state_dict，而不是完整的 checkpoint
        print("警告: 权重文件中未找到 'model' 键，尝试直接加载 state_dict。")
        model.load_state_dict(state_dict)

    model.to(device)
    print("加载模型成功!")
    # --- END: 修改权重加载逻辑 ---


    # ==================== START: 新增代码块 ==================== #
    #                                                             #
    #             统计并打印模型参数量 (Total & Trainable)          #
    #                                                             #
    # =========================================================== #
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 30)
    print(f"模型总参数量 (Total Parameters): {total_params:,}")
    print(f"可训练参数量 (Trainable Parameters): {trainable_params:,}")
    print("-" * 30)
    # ===================== END: 新增代码块 ===================== #


    # 自定义一个清晰的输出文件夹名
    suffix = "new_TUT_predictions"
    save_root = "./results/" + suffix

    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    with torch.no_grad():
        model.eval()
        # 使用tqdm来创建一个进度条
        pbar = tqdm(test_dl, desc="正在进行预测")
        for data in pbar:
            x = data["image"]
            target = data["label"] # 注意：这里的 target 实际上是掩码 (mask)

            if device.type != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()

            out = model(x)

            # --- START: 修正后的后处理和保存代码 ---

            # 1. 使用 Sigmoid 函数将模型输出转换为 0-1 之间的概率
            out = torch.sigmoid(out)

            # 2. 将 Pytorch Tensor 转换为 NumPy 数组，以便处理和保存
            target_np = target[0, 0, ...].cpu().numpy()
            prediction_np = out[0, 0, ...].cpu().numpy()

            # 从完整路径中提取不带后缀的文件名
            root_name = os.path.splitext(os.path.basename(data["A_paths"][0]))[0]

            # 3. 处理标签（Ground Truth）：确保它是 0 和 255 的二值图
            #    (原始标签是0和1，乘以255即可)
            target_to_save = (target_np * 255).astype(np.uint8)

            # 4. 处理预测结果：应用阈值（0.5）得到 0 和 1 的二值图，然后乘以 255
            prediction_binary = (prediction_np > 0.5).astype(np.uint8)
            prediction_to_save = prediction_binary * 255

            # 更新进度条的描述信息
            pbar.set_description(f"正在处理: {root_name}")

            # 保存修正后的清晰二值图像
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target_to_save)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), prediction_to_save)

            # --- END: 修正后的后处理和保存代码 ---

    print(f"\n预测完成！所有结果已保存至文件夹: {save_root}")