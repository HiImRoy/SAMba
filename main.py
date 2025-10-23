
import os
# 设置可见的 CUDA 设备，这在多 GPU 环境下很有用
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch.nn as nn

# --- 本地模块导入 ---
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval
from util.logger import get_logger
# 导入 PolyLR 学习率调度器
from mmengine.optim.scheduler.lr_scheduler import PolyLR
# 导入用于计算 FLOPs 的库
from thop import profile

# --- 辅助函数定义 --- #

def log_parameter_summary(model, log, args):
    """
    记录并打印模型的详细参数摘要，包括 FLOPs、参数量和模型大小。
    """
    log.info("--- 模型摘要 ---")

    # --- 1. 计算 FLOPs (浮点运算次数) ---
    try:
        # 创建一个符合模型输入尺寸的虚拟张量
        dummy_input = torch.randn(1, 3, args.load_height, args.load_width).to(next(model.parameters()).device)
        # 使用 thop.profile 计算 FLOPs 和参数量
        flops, params = profile(model, inputs=(dummy_input,))
        log.info(f"FLOPs: {flops / 1e9:.2f} G")
    except Exception as e:
        log.warning(f"无法计算 FLOPs: {e}")

    # --- 2. 计算参数数量 ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"总参数量: {total_params / 1e6:.2f} M")
    log.info(f"可训练参数量: {trainable_params / 1e6:.2f} M")
    if total_params > 0:
        log.info(f"可训练参数比例: {trainable_params / total_params * 100:.2f}%")

    # --- 3. 计算模型大小 (MB) ---
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2
    log.info(f"模型大小: {model_size_mb:.2f} MB")
    log.info("---------------------")

    # --- 4. 针对 SAMbaCrack 模型的详细参数分解 ---
    if args.model_name == 'SAMbaCrack':
        log.info("--- SAMbaCrack 详细参数分解 ---")
        
        # 定义需要分析的模块
        module_map = {
            "SAM Encoder (骨干网络, 含Adapter)": "sam_encoder",
            "SAVSS Input (Patch & Pos Embed)": ["savss_patch_embed", "savss_pos_embed"],
            "SAVSS Encoder (Mamba)": "mamba_encoder",
            "Fusion (FCM)": "fcms",
            "Decoder (U-Net)": "decoder",
            "SAM Downscale Adapters": "sam_adapters"
        }

        for name, attr_names in module_map.items():
            if not isinstance(attr_names, list):
                attr_names = [attr_names]
            
            module_total = 0
            module_trainable = 0
            found_module = False

            for attr_name in attr_names:
                if hasattr(model, attr_name):
                    found_module = True
                    module = getattr(model, attr_name)
                    if isinstance(module, nn.Parameter):
                        module_total += module.numel()
                        if module.requires_grad:
                            module_trainable += module.numel()
                    else:
                        module_total += sum(p.numel() for p in module.parameters())
                        module_trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)

            if found_module:
                trainable_percentage = (module_trainable / module_total * 100) if module_total > 0 else 0
                log.info(f"  - {name}:")
                log.info(f"    - 总参数: {module_total / 1e6:.3f}M")
                log.info(f"    - 可训练参数: {module_trainable / 1e6:.3f}M ({trainable_percentage:.2f}%)")

        log.info("----------------------------------------------------\n")

def save_plots(log_df, output_dir):
    """
    根据训练日志数据帧 (DataFrame)，生成并保存损失和评估指标的变化曲线图。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 绘制训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(log_df['epoch'], log_df['train_loss'], marker='o', linestyle='-', label='Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'loss_curve.png')
    plt.close()

    # 绘制所有评估指标的曲线
    metrics_to_plot = ['mIoU', 'Foreground_IoU', 'ODS', 'OIS', 'F1', 'Precision', 'Recall']
    plt.figure(figsize=(12, 8))
    for metric in metrics_to_plot:
        if metric in log_df.columns:
            plt.plot(log_df['epoch'], log_df[metric], marker='o', linestyle='-', label=metric)
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(output_dir / 'metrics_curve.png')
    plt.close()

def save_best_masks(model, device, args, output_dir, best_threshold):
    """
    使用最佳模型和最优阈值，在验证集上生成预测蒙版并与真实标签拼接后保存。
    """
    print(f"正在使用最优阈值 {best_threshold} 保存最佳模型的验证集预测蒙版至 {output_dir}...")
    model.eval()
    args.phase = 'test'
    args.batch_size = 1  # 一次处理一张图像
    test_dl = create_dataset(args)

    # 将 0-255 的整数阈值转换为 0-1 的浮点数阈值
    float_threshold = best_threshold / 255.0

    with torch.no_grad():
        for data in tqdm(test_dl, desc="生成最佳蒙版"):
            x, target = data["image"].to(device), data["label"].to(device)
            out = model(x)

            # 将 PyTorch 张量转换为 Numpy 数组用于可视化
            label_np = target[0, 0].cpu().numpy()
            
            # 将模型输出的 logits 通过 sigmoid 转换为概率图
            prob_map = torch.sigmoid(out)
            # 根据最优阈值进行二值化
            binary_map = (prob_map > float_threshold).float()
            pred_np_for_vis = binary_map[0, 0].cpu().numpy()

            # 转换为 8-bit 灰度图像以便保存
            label_vis = (255 * (label_np / (label_np.max() + 1e-8))).astype(np.uint8)
            pred_vis = (255 * pred_np_for_vis).astype(np.uint8)

            # 将灰度图转换为 BGR 彩色图以便绘制文字
            label_rgb = cv2.cvtColor(label_vis, cv2.COLOR_GRAY2BGR)
            pred_rgb = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

            # 将真实标签和预测结果水平拼接
            stitched_image = np.hstack((label_rgb, pred_rgb))

            # 在图像上添加 "Ground Truth" 和 "Prediction" 标签
            cv2.putText(stitched_image, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(stitched_image, 'Prediction', (label_rgb.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 保存拼接后的图像
            root_name = data["A_paths"][0].split("/")[-1]
            cv2.imwrite(str(output_dir / root_name), stitched_image)

def get_args_parser():
    """
    定义并解析所有命令行参数。参数被分为不同组以便于管理。
    """
    parser = argparse.ArgumentParser('SAMBA FOR CRACK', add_help=False)

    # --- 1. 模型与权重设置 (Model & Weights Settings) ---
    group = parser.add_argument_group('模型与权重设置 (Model & Weights Settings)')
    group.add_argument('--model_name', default='SAMbaCrack', type=str,
                       help="要使用的模型名称。")
    group.add_argument('--pretrained_weights', type=str, default='sam2_checkpoints/sam2.1_hiera_base_plus.pt',
                       help="预训练骨干网络（如 Hiera）的权重文件路径。")
    group.add_argument('--resume', default='', type=str,
                       help="从指定的 checkpoint 文件恢复训练，提供 .pth 文件的路径。")

    # --- 2. 损失函数设置 (Loss Function Settings) ---
    group = parser.add_argument_group('损失函数设置 (Loss Function Settings)')
    group.add_argument('--BCELoss_ratio', default=0.83, type=float,
                       help="总损失中二元交叉熵损失（BCE Loss）的权重。")
    group.add_argument('--DiceLoss_ratio', default=0.17, type=float,
                       help="总损失中 Dice 损失的权重。")

    # --- 3. 数据集与加载设置 (Dataset & Dataloader Settings) ---
    group = parser.add_argument_group('数据集与加载设置 (Dataset & Dataloader Settings)')
    group.add_argument('--dataset_path', default="data/DeepCrack", type=str,
                       help="数据集所在的根目录。")
    group.add_argument('--dataset_mode', type=str, default='crack',
                       help="要使用的数据集模式（例如 'crack'）。")
    group.add_argument('--batch_size_train', type=int, default=16,
                       help="训练时的批量大小。")
    group.add_argument('--batch_size_test', type=int, default=1,
                       help="测试/评估时的批量大小。")
    group.add_argument('--load_width', type=int, default=448,
                       help="加载图像时统一调整到的宽度。")
    group.add_argument('--load_height', type=int, default=448,
                       help="加载图像时统一调整到的高度。")
    group.add_argument('--num_threads', default=1, type=int,
                       help="数据加载时使用的工作线程数。")
    group.add_argument('--serial_batches', action='store_true',
                       help="是否禁用数据加载器的随机化，通常用于调试。")

    # --- 4. 训练超参数 (Training Hyperparameters) ---
    group = parser.add_argument_group('训练超参数 (Training Hyperparameters)')
    group.add_argument('--epochs', default=100, type=int,
                       help="总训练轮次。")
    group.add_argument('--start_epoch', default=0, type=int,
                       help="起始训练轮次，在恢复训练时会自动设置。")
    group.add_argument('--clip_grad_norm', default=1.0, type=float,
                       help="梯度裁剪的范数阈值，设为0则不进行梯度裁剪。")
    group.add_argument('--Norm_Type', default='GN', type=str,
                       help="模型中使用的归一化层类型（例如 'GN' for GroupNorm）。")

    # --- 5. 优化器与学习率调度器设置 (Optimizer & Scheduler Settings) ---
    group = parser.add_argument_group('优化器与学习率调度器设置 (Optimizer & Scheduler Settings)')
    group.add_argument('--lr', default=1e-4, type=float,
                       help="优化器的初始学习率。")
    group.add_argument('--min_lr', default=1e-6, type=float,
                       help="学习率调度器允许的最低学习率。")
    group.add_argument('--lr_scheduler', type=str, default='PolyLR',
                       help="要使用的学习率调度器类型（例如 'PolyLR'）。")
    group.add_argument('--lr_backbone_multiplier', default=0.1, type=float,
                       help="应用于微调部分（如 Adapter）参数的学习率乘子。")
    group.add_argument('--weight_decay', default=0.01, type=float,
                       help="AdamW 优化器的权重衰减系数。")
    group.add_argument('--sgd', action='store_true',
                       help="【已弃用】是否使用 SGD 优化器。")
    group.add_argument('--lr_drop', default=30, type=int,
                       help="【已弃用】StepLR 调度器的学习率下降轮次。")

    # --- 6. 路径与输出设置 (Path & Output Settings) ---
    group = parser.add_argument_group('路径与输出设置 (Path & Output Settings)')
    group.add_argument('--output_dir', default='./results/samba_v19.1',
                       help="所有实验结果（日志、权重、图像）的根目录。")

    # --- 7. 环境与杂项设置 (Environment & Miscellaneous Settings) ---
    group = parser.add_argument_group('环境与杂项设置 (Environment & Miscellaneous Settings)')
    group.add_argument('--device', default='cuda',
                       help="训练和推理使用的设备 ('cuda' or 'cpu')。")
    group.add_argument('--seed', default=42, type=int,
                       help="为CPU、Numpy和PyTorch设置随机种子以保证可复现性。")
    group.add_argument('--phase', type=str, default='train',
                       help="当前运行阶段 ('train' or 'test')，由脚本内部管理。")
    
    return parser

def main(args):
    """
    主执行函数，负责整个训练和评估流程。
    """
    # --- 1. 设置输出路径和日志记录器 ---
    if args.resume:
        # 如果是恢复训练，则使用现有实验目录
        output_dir = Path(args.resume).parent.parent
        exp_name = output_dir.name
    else:
        # 否则，创建一个新的、带时间戳的实验目录
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        dataset_name = Path(args.dataset_path).name
        exp_name = f"{cur_time}_{args.model_name}_{dataset_name}"
        output_dir = Path(args.output_dir) / exp_name

    # 创建权重、蒙版、图表和原始预测子目录
    weights_dir = output_dir / 'weights'
    masks_dir = output_dir / 'best_epoch_masks'
    plots_dir = output_dir / 'plots'
    raw_preds_base_dir = output_dir / 'raw_predictions_by_epoch' # 新增：用于存放所有 epoch 的原始预测图
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    raw_preds_base_dir.mkdir(exist_ok=True) # 新增：创建原始预测图的父目录

    # 初始化日志记录器
    log = get_logger(output_dir, 'experiment_log')
    
    if args.resume:
        log.info("\n" + "="*20 + " 恢复训练 " + "="*20 + "\n")
        
    log.info(f"实验开始: {exp_name}")
    log.info(f"所有结果将保存至: {output_dir}")
    log.info("--- 超参数 ---")
    for arg, value in sorted(vars(args).items()):
        log.info(f"{arg}: {value}")
    log.info("-----------------------\n")

    # --- 2. 环境设置 ---
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- 3. 构建模型和损失函数 ---
    log.info("--- 正在构建模型 ---")
    model, criterion = build_model(args)
    model.to(device)

    # 如果不是恢复训练，则尝试加载预训练权重
    if hasattr(model, 'init_weights') and not args.resume:
        log.info("正在初始化权重... 如果提供了预训练权重路径，则会加载。")
        model.init_weights(args.pretrained_weights)

    # 打印模型的详细参数信息
    log_parameter_summary(model, log, args)

    # 记录 SAMbaCrack 的特定模型超参数
    if args.model_name == 'SAMbaCrack':
        log.info("--- SAMbaCrack 模型内部超参数 ---")
        try:
            log.info(f"sam_dims: {model.sam_dims}")
            log.info(f"savss_dims: {model.savss_dims}")
            log.info(f"hiera_depths: {model.hiera_depths}")
            log.info(f"savss_depths: {model.savss_depths}") # 新增行
            log.info(f"hiera_num_heads: {model.hiera_num_heads}")
            log.info(f"hiera_patch_size: {model.hiera_patch_size}")
            log.info(f"savss_patch_size: {model.savss_patch_size}")
            log.info(f"fcm_output_dims: {model.fcm_output_dims}")
        except AttributeError as e:
            log.warning(f"无法记录部分模型超参数，因为模型实例上缺少该属性: {e}")
        log.info("------------------------------------\n")

    # --- 4. 创建数据加载器 ---
    args.phase = 'train'
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    log.info(f'训练集图片数量 = {len(train_dataLoader.dataset)}')
    log.info(f'训练集批次数 = {len(train_dataLoader)}')

    # --- 5. 设置优化器和学习率调度器 ---
    # 为需要微调的参数（Adapter）和从头训练的参数设置不同的学习率
    if args.model_name == 'SAMbaCrack' and hasattr(model, 'sam_encoder'):
        log.info("为 SAMbaCrack 创建特定的参数组 (微调 vs 从头训练)。")
        
        # 识别出所有属于微调部分的参数（即 sam_encoder 内部的可训练参数）
        finetune_params = [p for p in model.sam_encoder.parameters() if p.requires_grad]
        finetune_param_ids = set(map(id, finetune_params))

        # 从头训练的参数是所有可训练参数中，不属于微调部分的其他参数
        scratch_params = [p for p in model.parameters() if p.requires_grad and id(p) not in finetune_param_ids]

        param_dicts = [
            {"params": scratch_params, "lr": args.lr},
            {"params": finetune_params, "lr": args.lr * args.lr_backbone_multiplier},
        ]
        log.info(f"优化器分组: {len(scratch_params)} 个参数从头训练, {len(finetune_params)} 个参数进行微调。")
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        # 对于其他模型或一般情况，使用单一学习率
        log.info(f"为 {args.model_name} 创建单一参数组的优化器。")
        param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # 初始化学习率调度器
    lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)

    # --- 6. 恢复训练 (如果需要) ---
    if args.resume:
        if os.path.isfile(args.resume):
            log.info(f"--- 正在从 checkpoint 恢复: {args.resume} ---")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            log.info(f"--- 恢复成功。将从 epoch {args.start_epoch} 开始 ---")
        else:
            log.warning(f"在 {args.resume} 未找到 checkpoint 文件。将从头开始训练。")

    # --- 7. 开始主训练循环 ---
    log.info("--- 开始训练 ---")
    start_time = time.time()
    best_mIoU = 0.0
    best_metrics_from_best_mIoU_epoch = {}
    
    # 加载或初始化训练日志
    log_csv_path = output_dir / 'training_log.csv'
    if args.resume and log_csv_path.exists():
        log.info(f"从 {log_csv_path} 加载之前的日志")
        log_df = pd.read_csv(log_csv_path)
        log_data = log_df.to_dict('records')
        if 'mIoU' in log_df.columns and not log_df['mIoU'].empty:
            best_mIoU = log_df['mIoU'].max()
            log.info(f"找到之前的最佳 mIoU: {best_mIoU:.4f}")
    else:
        log_data = []

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        log.info(f"\n===== Epoch {epoch}/{args.epochs - 1} ======")
        
        # --- 7a. 训练一个 Epoch ---
        train_stats = train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log)
        lr_scheduler.step()

        # 记录 GPU 显存使用情况
        if device.type == 'cuda':
            log.info(f"GPU 显存: 已分配={torch.cuda.memory_allocated(0)/1024**2:.1f}MB, 峰值={torch.cuda.max_memory_allocated(0)/1024**2:.1f}MB")
            torch.cuda.reset_peak_memory_stats(0)

        # --- 7b. 在验证集上进行评估 ---
        # 修改：将每个 epoch 的原始预测图保存到 raw_preds_base_dir 下的子文件夹
        temp_eval_dir = raw_preds_base_dir / f'epoch_{epoch}' 
        temp_eval_dir.mkdir(exist_ok=True)

        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        with torch.no_grad():
            model.eval()
            for data in tqdm(test_dl, desc=f"测试 Epoch {epoch}"):
                images, targets = data["image"].to(device), data["label"].cpu().numpy()
                outputs = model(images)
                prob_maps = torch.sigmoid(outputs)

                # 遍历批次中的每一张图片并保存
                for i in range(images.size(0)):
                    prob_map_uint8 = (prob_maps[i, 0] * 255).cpu().numpy().astype(np.uint8)
                    target_uint8 = (targets[i, 0] * 255).astype(np.uint8)
                    
                    # 保存原始预测图和标签图以供 `eval` 函数使用
                    root_name = data["A_paths"][i].split("/")[-1][:-4]
                    cv2.imwrite(str(temp_eval_dir / f"{root_name}_pre.png"), prob_map_uint8)
                    cv2.imwrite(str(temp_eval_dir / f"{root_name}_lab.png"), target_uint8)

        # 调用评估脚本计算各项指标
        current_epoch_metrics = eval(log, str(temp_eval_dir), epoch)

        # --- 7c. 打印、记录和保存结果 ---
        print(f"\n" + "-"*25 + f" Epoch {epoch} 总结 " + "-"*25)
        print(f"  - 训练损失: {train_stats['loss']:.4f}")
        print(f"  - mIoU:       {current_epoch_metrics.get('mIoU', 0.0):.4f}")
        print(f"  - 前景 IoU:   {current_epoch_metrics.get('Foreground_IoU', 0.0):.4f}")
        print(f"  - ODS (F1):   {current_epoch_metrics.get('ODS', 0.0):.4f}")
        print(f"  - OIS (F1):   {current_epoch_metrics.get('OIS', 0.0):.4f}")
        print(f"  - 查准率:     {current_epoch_metrics.get('Precision', 0.0):.4f}")
        print(f"  - 查全率:     {current_epoch_metrics.get('Recall', 0.0):.4f}")
        print(f"  - 最佳阈值:   {current_epoch_metrics.get('best_threshold', -1)}")
        print("-" * 70 + f"\n")

        log.info(f"Epoch {epoch} 验证集指标: {current_epoch_metrics}")

        # 将当前 epoch 的结果添加到日志列表中
        log_entry = {'epoch': epoch, 'train_loss': train_stats['loss'], **current_epoch_metrics}
        log_data.append(log_entry)

        # 准备 checkpoint 数据
        checkpoint_payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        # 保存最新的 checkpoint
        utils.save_on_master(checkpoint_payload, weights_dir / 'checkpoint_last.pth')

        # 如果当前 mIoU 是历史最佳，则保存为最佳 checkpoint
        if current_epoch_metrics.get('mIoU', 0) > best_mIoU:
            best_mIoU = current_epoch_metrics.get('mIoU', 0)
            best_metrics_from_best_mIoU_epoch = current_epoch_metrics
            log.info(f"*** 在 epoch {epoch} 发现新的最佳 mIoU: {best_mIoU:.4f}！正在保存最佳模型和蒙版... ***")

            utils.save_on_master(checkpoint_payload, weights_dir / 'checkpoint_best.pth')

            # 使用最佳模型和该 epoch 找到的最优阈值来保存可视化结果
            best_threshold_for_saving = best_metrics_from_best_mIoU_epoch.get('best_threshold', 127)
            save_best_masks(model, device, args, masks_dir, best_threshold_for_saving)

        log.info(f"Epoch {epoch} 在 {(time.time() - epoch_start_time):.2f}s 内完成。当前最佳 mIoU: {best_mIoU:.4f}")

    # --- 8. 训练结束，总结并保存最终结果 ---
    total_time = time.time() - start_time
    log.info(f"--- 训练在 {datetime.timedelta(seconds=int(total_time))} 内完成 ---")

    # 打印最佳模型的性能摘要
    if best_metrics_from_best_mIoU_epoch:
        final_report = {
            '最佳 Epoch': best_metrics_from_best_mIoU_epoch.get('epoch'),
            'mIoU': best_metrics_from_best_mIoU_epoch.get('mIoU'),
            '前景 IoU': best_metrics_from_best_mIoU_epoch.get('Foreground_IoU'),
            'ODS': best_metrics_from_best_mIoU_epoch.get('ODS'),
            'OIS': best_metrics_from_best_mIoU_epoch.get('OIS'),
            '查准率': best_metrics_from_best_mIoU_epoch.get('Precision', 0),
            '查全率': best_metrics_from_best_mIoU_epoch.get('Recall', 0),
            'F1 分数 (来自最佳 Epoch)': best_metrics_from_best_mIoU_epoch.get('F1', 0.0),
            '最佳阈值': best_metrics_from_best_mIoU_epoch.get('best_threshold', -1)
        }
        log.info("--- 最佳模型性能 (基于最高 mIoU) ---")
        for key, value in final_report.items():
            log.info(f"  - {key}: {value}")
    else:
        log.info("训练期间未找到最佳模型。")

    # 将完整的训练日志保存为 CSV 文件，并绘制曲线图
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(output_dir / 'training_log.csv', index=False)
    save_plots(log_df, plots_dir)
    log.info(f"\n训练日志已保存至 {output_dir / 'training_log.csv'}")
    log.info(f"损失和指标曲线图已保存在 {plots_dir}")
    log.info("--- 实验完成 ---")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser('SAMBA FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    # 运行主函数
    main(args)
