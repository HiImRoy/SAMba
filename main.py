import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn  # 引入 nn 以便在 custom_ops 中使用
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Local Imports
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval
from util.logger import get_logger
# --- RE-ADDED: PolyLR for stable training ---
from mmengine.optim.scheduler.lr_scheduler import PolyLR
# --- [NEW] Import for FLOPs calculation ---
from thop import profile


# --- [FIXED] Helper function for FLOPs calculation ---
def count_mha(m, x, y):
    """自定义 nn.MultiheadAttention 的 FLOPs 计数器来绕过 thop 的 bug。"""
    # 简化计算，主要考虑 Q, K, V 的线性投影和输出投影
    # x[0] 是 query, shape (B, L, C)
    B, L, C = x[0].shape
    # 假设 in_proj_weight 存在
    if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
        # q, k, v 的合并投影
        m.total_ops += torch.DoubleTensor([B * L * C * C * 3])
    # 输出投影
    if hasattr(m, 'out_proj') and m.out_proj is not None:
        m.total_ops += torch.DoubleTensor([B * L * C * C])


# --- [FIXED] Helper Functions --- #
def log_parameter_summary(model, log, args):
    """[重构后] 的日志函数，用于记录模块化的 SAMbaCrack-AF 模型参数。"""
    log.info("--- Model Summary ---")

    # --- 1. FLOPs Calculation ---
    try:
        dummy_input = torch.randn(1, 3, args.load_height, args.load_width).to(next(model.parameters()).device)
        # 为 MultiheadAttention 提供自定义的 FLOPs 计数器以解决报错
        custom_ops = {nn.MultiheadAttention: count_mha}
        flops, params = profile(model, inputs=(dummy_input,), custom_ops=custom_ops)
        log.info(f"FLOPs: {flops / 1e9:.2f} G")
    except Exception as e:
        log.warning(f"无法计算 FLOPs: {e}")

    # --- 2. Parameter Count ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Params: {total_params / 1e6:.2f} M")
    log.info(f"Trainable Params: {trainable_params / 1e6:.2f} M")
    if total_params > 0:
        log.info(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%")

    # --- 3. Model Size ---
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 ** 2
    log.info(f"Model Size: {model_size_mb:.2f} MB")
    log.info("---------------------")

    # --- 4. [FIXED] Detailed Breakdown for SAMbaCrack-AF ---
    if args.model_name == 'SAMbaCrack' and hasattr(model, 'backbone'):
        log.info("--- Detailed Parameter Breakdown (SAMbaCrack-AF) ---")

        # 定义新的模块化结构
        # model 现在是 SAMbaCrackAF，其子模块是 backbone, neck, head
        module_map = {
            "Backbone (SAM/Hiera Frozen)": model.backbone.sam_encoder,
            "Backbone (SAM Refiners)": model.backbone.refiners,
            "Backbone (SAM Adapters)": model.backbone.adapters,
            "Backbone (SAVSS Embed)": model.backbone.savss_patch_embed,
            "Backbone (SAVSS Encoder)": model.backbone.mamba_encoder,
            "Backbone (SAVSS Decoders)": model.backbone.savss_decoder_heads,
            "Neck (Fusion)": model.neck,
            "Head (Decoder)": model.head
        }

        for name, module in module_map.items():
            if module is not None:
                module_total = sum(p.numel() for p in module.parameters())
                module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                trainable_percentage = (module_trainable / module_total * 100) if module_total > 0 else 0

                log.info(f"  - {name}:")
                log.info(f"    - Total params: {module_total / 1e6:.3f}M")
                log.info(f"    - Trainable params: {module_trainable / 1e6:.3f}M ({trainable_percentage:.2f}%)")

        log.info("----------------------------------------------------\n")


def save_plots(log_df, output_dir):
    """Generates and saves plots for loss and metrics."""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(12, 6))
    plt.plot(log_df['epoch'], log_df['train_loss'], marker='o', linestyle='-', label='Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'loss_curve.png')
    plt.close()

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
    """Saves stitched prediction and label masks for the best model using the optimal threshold."""
    print(f"Saving validation masks for the best model to {output_dir} (using ODS threshold: {best_threshold})...")
    model.eval()
    args.phase = 'test'
    args.batch_size = 1
    test_dl = create_dataset(args)

    float_threshold = best_threshold / 255.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dl, desc="Generating best masks")):
            if isinstance(data, (list, tuple)):
                if len(data) == 3:
                    x, target, path = data
                    root_name = os.path.basename(path[0]).split('.')[0]
                else:
                    x, target = data
                    root_name = f"best_mask_{i}"
                x = x.to(device)
                target = target.to(device)
            else: # Original dataset structure
                x, target = data["image"].to(device), data["label"].to(device)
                root_name = data["A_paths"][0].split("/")[-1]

            out = model(x)

            if target.dim() == 3:
                target = target.unsqueeze(1)

            label_np = target[0, 0].cpu().numpy()

            prob_map = torch.sigmoid(out)
            binary_map = (prob_map > float_threshold).float()
            pred_np_for_vis = binary_map[0, 0].cpu().numpy()

            label_vis = (255 * (label_np / (label_np.max() + 1e-8))).astype(np.uint8)
            pred_vis = (255 * pred_np_for_vis).astype(np.uint8)

            label_rgb = cv2.cvtColor(label_vis, cv2.COLOR_GRAY2BGR)
            pred_rgb = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

            stitched_image = np.hstack((label_rgb, pred_rgb))

            cv2.putText(stitched_image, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(stitched_image, 'Prediction', (label_rgb.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            cv2.imwrite(str(output_dir / f"{root_name}.png"), stitched_image)


def get_args_parser():
    parser = argparse.ArgumentParser('SAMBA FOR CRACK', add_help=False)
    parser.add_argument('--model_name', default='SAMbaCrack', type=str)
    parser.add_argument('--pretrained_weights', type=str, default='sam2_checkpoints/sam2.1_hiera_base_plus.pt',
                        help='Path to the pretrained Hiera weights.')

    parser.add_argument('--BCELoss_ratio', default=0.83, type=float,
                        help="Weight for BCE Loss in the total loss function.")
    parser.add_argument('--DiceLoss_ratio', default=0.17, type=float,
                        help="Weight for Dice Loss in the total loss function.")

    parser.add_argument('--Norm_Type', default='GN', type=str)
    parser.add_argument('--dataset_path', default="data/CrackTree260")
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=1)

    parser.add_argument('--lr_scheduler', type=str, default='PolyLR', help='LR scheduler to use.')
    parser.add_argument('--lr', default=1e-4, type=float, help="The initial learning rate for PolyLR.")

    parser.add_argument('--clip_grad_norm', default=1.0, type=float,
                        help="Gradient clipping norm value (0 for no clipping).")

    parser.add_argument('--lr_backbone_multiplier', default=0.1, type=float,
                        help="Multiplier for the learning rate of fine-tuned parts (e.g., SAM adapters).")

    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)

    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume training from.')

    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='./results/FPN', help='Root directory for all outputs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load_width', type=int, default=448)
    parser.add_argument('--load_height', type=int, default=448)
    return parser


def main(args):
    if args.resume:
        output_dir = Path(args.resume).parent.parent
        exp_name = output_dir.name
    else:
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        dataset_name = Path(args.dataset_path).name
        exp_name = f"{cur_time}_{args.model_name}_{dataset_name}"
        output_dir = Path(args.output_dir) / exp_name

    weights_dir = output_dir / 'weights'
    masks_dir = output_dir / 'best_epoch_masks'
    plots_dir = output_dir / 'plots'

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    log = get_logger(output_dir, 'experiment_log')

    if args.resume:
        log.info("\n" + "=" * 20 + " RESUMING TRAINING " + "=" * 20 + "\n")

    log.info(f"Experiment started: {exp_name}")
    log.info(f"Results will be saved to: {output_dir}")
    log.info("--- Hyperparameters ---")
    for arg, value in sorted(vars(args).items()):
        log.info(f"{arg}: {value}")
    log.info("-----------------------\n")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    log.info("--- Building Model ---")
    model, criterion = build_model(args)
    model.to(device)

    if hasattr(model, 'init_weights') and not args.resume:
        log.info(f"Initializing weights... Will use pretrained weights if path is provided.")
        model.init_weights(args.pretrained_weights)

    log_parameter_summary(model, log, args)

    args.phase = 'train'
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    log.info(f'The number of training images = {len(train_dataLoader.dataset)}')
    log.info(f'Number of training batches = {len(train_dataLoader)}')

    if args.model_name == 'SAMbaCrack' and hasattr(model, 'backbone'):
        log.info("Creating optimizer with separate LRs for SAMbaCrack-AF (finetuning vs. scratch).")
        finetune_param_ids = set(map(id, model.backbone.refiners.parameters())) | set(
            map(id, model.backbone.adapters.parameters()))
        scratch_params = [p for p in model.parameters() if p.requires_grad and id(p) not in finetune_param_ids]
        finetune_params = [p for p in model.parameters() if p.requires_grad and id(p) in finetune_param_ids]
        param_dicts = [
            {"params": scratch_params, "lr": args.lr},
            {"params": finetune_params, "lr": args.lr * args.lr_backbone_multiplier},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        log.info(f"Creating optimizer with a single parameter group for {args.model_name}.")
        param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            log.info(f"--- Resuming training from checkpoint: {args.resume} ---")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            log.info(f"--- Resumed successfully. Starting from epoch {args.start_epoch} ---")
        else:
            log.warning(f"Checkpoint file not found at {args.resume}. Starting from scratch.")

    log.info("--- Starting Training ---")
    start_time = time.time()
    best_mIoU = 0.0
    best_metrics_from_best_mIoU_epoch = {}

    log_csv_path = output_dir / 'training_log.csv'
    if args.resume and log_csv_path.exists():
        log.info(f"Loading previous logs from {log_csv_path}")
        log_df = pd.read_csv(log_csv_path)
        log_data = log_df.to_dict('records')
        if 'mIoU' in log_df.columns and not log_df['mIoU'].empty:
            best_mIoU = log_df['mIoU'].max()
            log.info(f"Found previous best mIoU: {best_mIoU:.4f}")
    else:
        log_data = []

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        log.info(f"\n===== Epoch {epoch}/{args.epochs - 1} =====")
        train_stats = train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log)
        lr_scheduler.step()

        if device.type == 'cuda':
            log.info(
                f"GPU Memory: Allocated={torch.cuda.memory_allocated(0) / 1024 ** 2:.1f}MB, Peak={torch.cuda.max_memory_allocated(0) / 1024 ** 2:.1f}MB")
            torch.cuda.reset_peak_memory_stats(0)

        temp_eval_dir = output_dir / f'epoch_{epoch}_raw_preds'
        temp_eval_dir.mkdir(exist_ok=True)

        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        with torch.no_grad():
            model.eval()
            # --- [FIXED] Final fix for validation loop to handle all data formats and save correctly ---
            for i, data in enumerate(tqdm(test_dl, desc=f"Testing Epoch {epoch}")):
                if isinstance(data, (list, tuple)):
                    if len(data) == 3:
                        x, target_tensor, path = data
                        root_name = os.path.basename(path[0]).split('.')[0]
                    else:
                        x, target_tensor = data
                        root_name = f"val_image_{i}"
                    x = x.to(device)
                    target_np = target_tensor.cpu().numpy()
                    # [FIXED] Correctly get the 2D mask and ensure it's uint8 [0, 255]
                    target_to_save = target_np[0].astype(np.uint8)
                else:
                    x = data["image"].to(device)
                    target_np = data["label"].cpu().numpy()
                    root_name = data["A_paths"][0].split("/")[-1][0:-4]
                    target_to_save = (target_np[0, 0] * 255).astype(np.uint8)

                out = model(x)
                prob_map_0_1 = torch.sigmoid(out)
                prob_map_uint8 = (prob_map_0_1[0, 0] * 255).cpu().numpy().astype(np.uint8)
                
                # Save prediction and label with the correct names for eval.py
                cv2.imwrite(str(temp_eval_dir / f"{root_name}_pre.png"), prob_map_uint8)
                cv2.imwrite(str(temp_eval_dir / f"{root_name}_lab.png"), target_to_save)

        current_epoch_metrics = eval(log, str(temp_eval_dir), epoch)

        print(f"\n" + "-" * 25 + f" Epoch {epoch} Summary " + "-" * 25)
        print(f"  - Train Loss: {train_stats['loss']:.4f}")
        print(f"  - mIoU:       {current_epoch_metrics.get('mIoU', 0.0):.4f}")
        print(f"  - Fg IoU:     {current_epoch_metrics.get('Foreground_IoU', 0.0):.4f}")
        print(f"  - ODS (F1):   {current_epoch_metrics.get('ODS', 0.0):.4f}")
        print(f"  - OIS (F1):   {current_epoch_metrics.get('OIS', 0.0):.4f}")
        print(f"  - Precision:  {current_epoch_metrics.get('Precision', 0.0):.4f}")
        print(f"  - Recall:     {current_epoch_metrics.get('Recall', 0.0):.4f}")
        print(f"  - Best Thresh:{current_epoch_metrics.get('best_threshold', -1)}")
        print("-" * 70 + f"\n")

        log.info(f"Epoch {epoch} Validation Metrics: {current_epoch_metrics}")

        log_entry = {'epoch': epoch, 'train_loss': train_stats['loss'], **current_epoch_metrics}
        log_data.append(log_entry)

        checkpoint_payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        utils.save_on_master(checkpoint_payload, weights_dir / 'checkpoint_last.pth')

        if current_epoch_metrics.get('mIoU', 0) > best_mIoU:
            best_mIoU = current_epoch_metrics.get('mIoU', 0)
            best_metrics_from_best_mIoU_epoch = current_epoch_metrics
            log.info(f"*** New best mIoU: {best_mIoU:.4f} at epoch {epoch}! Saving best model and masks. ***")

            utils.save_on_master(checkpoint_payload, weights_dir / 'checkpoint_best.pth')

            best_threshold_for_saving = best_metrics_from_best_mIoU_epoch.get('best_threshold', 127)
            save_best_masks(model, device, args, masks_dir, best_threshold_for_saving)

        log.info(
            f"Epoch {epoch} finished in {(time.time() - epoch_start_time):.2f}s. Current best mIoU: {best_mIoU:.4f}")

    total_time = time.time() - start_time
    log.info(f"--- Training Finished in {datetime.timedelta(seconds=int(total_time))} ---")

    if best_metrics_from_best_mIoU_epoch:
        best_precision = best_metrics_from_best_mIoU_epoch.get('Precision', 0)
        best_recall = best_metrics_from_best_mIoU_epoch.get('Recall', 0)

        final_report = {
            'Best_Epoch': best_metrics_from_best_mIoU_epoch.get('epoch'),
            'mIoU': best_metrics_from_best_mIoU_epoch.get('mIoU'),
            'Foreground_IoU': best_metrics_from_best_mIoU_epoch.get('Foreground_IoU'),
            'ODS': best_metrics_from_best_mIoU_epoch.get('ODS'),
            'OIS': best_metrics_from_best_mIoU_epoch.get('OIS'),
            'Precision': best_precision,
            'Recall': best_recall,
            'F1_Score (from Best Epoch)': best_metrics_from_best_mIoU_epoch.get('F1', 0.0),
            'Best_Threshold': best_metrics_from_best_mIoU_epoch.get('best_threshold', -1)
        }

        log.info("--- Best Model Performance (based on max mIoU) ---")
        for key, value in final_report.items():
            log.info(f"  - {key}: {value}")
    else:
        log.info("No best model was found during training.")

    log_df = pd.DataFrame(log_data)
    log_df.to_csv(output_dir / 'training_log.csv', index=False)
    save_plots(log_df, plots_dir)
    log.info(f"\nTraining log saved to {output_dir / 'training_log.csv'}")
    log.info(f"Loss and metric plots saved in {plots_dir}")
    log.info("--- Experiment Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAMBA FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
