
import os
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

# Local Imports
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval
from util.logger import get_logger
# REMOVED: from mmengine.optim.scheduler.lr_scheduler import PolyLR

# --- Helper Functions --- #

def log_parameter_summary(model, log):
    """Logs a detailed breakdown of model parameters."""
    log.info("--- Model Parameter Breakdown ---")

    # Overall summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info(f"Total parameters: {total_params / 1e6:.2f}M")
    log.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    if total_params > 0:
        log.info(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%")
    log.info("-----------------------------")

    # --- Manual calculation for SAVSS Input Stage ---
    savss_input_total = sum(p.numel() for p in model.savss_patch_embed.parameters()) + model.savss_pos_embed.numel()
    savss_input_trainable = sum(p.numel() for p in model.savss_patch_embed.parameters() if p.requires_grad) + (model.savss_pos_embed.numel() if model.savss_pos_embed.requires_grad else 0)
    
    trainable_percentage = 0
    if savss_input_total > 0:
        trainable_percentage = (savss_input_trainable / savss_input_total) * 100

    log.info(f"  - SAVSS Input (PatchEmbed + PosEmbed):")
    log.info(f"    - Total params: {savss_input_total / 1e6:.3f}M")
    log.info(f"    - Trainable params: {savss_input_trainable / 1e6:.3f}M ({trainable_percentage:.2f}%)")

    # Breakdown by other modules
    module_map = {
        "SAM Encoder (Frozen)": model.sam_encoder,
        "SAM Refiners": model.refiners,
        "SAM Adapters": model.adapters,
        "SAVSS Encoder (Mamba)": model.mamba_encoder,
        "Fusion (HOACM)": model.hoacms,
        "Decoder Projections": model.projections,
        "Decoder (MFS)": model.decoder
    }

    for name, module in module_map.items():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        trainable_percentage = 0
        if module_total > 0:
            trainable_percentage = (module_trainable / module_total) * 100

        log.info(f"  - {name}:")
        log.info(f"    - Total params: {module_total / 1e6:.3f}M")
        log.info(f"    - Trainable params: {module_trainable / 1e6:.3f}M ({trainable_percentage:.2f}%)")

    log.info("------------------------------------\n")

def save_plots(log_df, output_dir):
    """Generates and saves plots for loss and metrics."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(log_df['epoch'], log_df['train_loss'], marker='o', linestyle='-', label='Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'loss_curve.png')
    plt.close()

    # Plot Metrics
    metrics_to_plot = ['mIoU', 'ODS', 'OIS', 'F1', 'Precision', 'Recall']
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

def save_best_masks(model, device, args, output_dir):
    """Saves stitched prediction and label masks for the best model."""
    print(f"Saving validation masks for the best model to {output_dir}...")
    model.eval()
    args.phase = 'test'
    args.batch_size = 1
    test_dl = create_dataset(args)
    
    with torch.no_grad():
        for data in tqdm(test_dl, desc="Generating best masks"):
            x, target = data["image"].to(device), data["label"].to(device)
            out = model(x)

            label_np = target[0, 0].cpu().numpy()
            pred_np = torch.sigmoid(out)[0, 0].cpu().numpy()

            label_vis = (255 * (label_np / (label_np.max() + 1e-8))).astype(np.uint8)
            pred_vis = (255 * pred_np).astype(np.uint8)

            label_rgb = cv2.cvtColor(label_vis, cv2.COLOR_GRAY2BGR)
            pred_rgb = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

            stitched_image = np.hstack((label_rgb, pred_rgb))

            cv2.putText(stitched_image, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(stitched_image, 'Prediction', (label_rgb.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            root_name = data["A_paths"][0].split("/")[-1]
            cv2.imwrite(str(output_dir / root_name), stitched_image)

def get_args_parser():
    # The description is updated here
    parser = argparse.ArgumentParser('SAMBA FOR CRACK', add_help=False)
    parser.add_argument('--model_name', default='SAMbaCrack', type=str)
    parser.add_argument('--pretrained_weights', type=str, default='sam2_checkpoints/sam2.1_hiera_small.pt', help='Path to the pretrained Hiera weights.')
    
    # Balanced loss ratios for faster convergence
    parser.add_argument('--BCELoss_ratio', default=0.5, type=float, help="Weight for BCE Loss in the total loss function.")
    parser.add_argument('--DiceLoss_ratio', default=0.5, type=float, help="Weight for Dice Loss in the total loss function.")
    
    parser.add_argument('--Norm_Type', default='GN', type=str)
    parser.add_argument('--dataset_path', default="data/crack500")
    # Batch size
    parser.add_argument('--batch_size_train', type=int, default=8)
    parser.add_argument('--batch_size_test', type=int, default=8)

    # --- MODIFIED: Switched to OneCycleLR --- #
    parser.add_argument('--lr_scheduler', type=str, default='OneCycleLR', help='LR scheduler to use.')
    parser.add_argument('--lr', default=8e-4, type=float, help="The maximum learning rate for OneCycleLR.")
    
    # --- ADDED: Gradient Clipping --- #
    parser.add_argument('--clip_grad_norm', default=1.0, type=float, help="Gradient clipping norm value (0 for no clipping).")

    # Added argument for differential learning rate
    parser.add_argument('--lr_backbone_multiplier', default=0.1, type=float, help="Multiplier for the learning rate of fine-tuned parts (e.g., SAM adapters).")
    
    parser.add_argument('--min_lr', default=1e-6, type=float) # This is now used by PolyLR if you switch back
    parser.add_argument('--weight_decay', default=0.01, type=float)
    # Epoch数
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)

    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='./results', help='Root directory for all outputs')
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
    # 1. SETUP DIRECTORY STRUCTURE
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

    # 2. SETUP LOGGING
    log = get_logger(output_dir, 'experiment_log')
    log.info(f"Experiment started: {exp_name}")
    log.info(f"Results will be saved to: {output_dir}")
    log.info("--- Hyperparameters ---")
    for arg, value in sorted(vars(args).items()):
        log.info(f"{arg}: {value}")
    log.info("-----------------------\n")

    # 3. SETUP DEVICE & SEED
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 4. BUILD MODEL, LOAD WEIGHTS, & CREATE DATASET
    log.info("--- Building Model ---")
    model, criterion = build_model(args)
    model.to(device)

    # --- LOAD PRETRAINED WEIGHTS ---
    if hasattr(model, 'init_weights'):
        log.info(f"Initializing weights... Will use pretrained weights if path is provided.")
        model.init_weights(args.pretrained_weights)

    # --- LOG PARAMETER BREAKDOWN ---
    log_parameter_summary(model, log)

    args.phase = 'train'
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    log.info(f'The number of training images = {len(train_dataLoader.dataset)}')
    log.info(f'Number of training batches = {len(train_dataLoader)}')

    # 5. SETUP OPTIMIZER & SCHEDULER
    log.info("--- Setting up Optimizer with Differential Learning Rate ---")

    # Group parameters for differential learning rate
    # Fine-tune parameters are from the SAM refiners and adapters
    finetune_param_ids = set(map(id, model.refiners.parameters())) | set(map(id, model.adapters.parameters()))

    # Scratch parameters are all other trainable parameters
    scratch_params = [p for p in model.parameters() if p.requires_grad and id(p) not in finetune_param_ids]
    finetune_params = [p for p in model.parameters() if p.requires_grad and id(p) in finetune_param_ids]

    param_dicts = [
        {"params": scratch_params, "lr": args.lr},
        {"params": finetune_params, "lr": args.lr * args.lr_backbone_multiplier},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Log the parameter groups for verification
    num_scratch_params = sum(p.numel() for p in scratch_params)
    num_finetune_params = sum(p.numel() for p in finetune_params)
    log.info(f"Optimizer: AdamW with base_lr={args.lr}, weight_decay={args.weight_decay}")
    log.info(f"  - Scratch Group ({num_scratch_params/1e6:.3f}M params) lr: {args.lr}")
    log.info(f"  - Finetune Group ({num_finetune_params/1e6:.3f}M params) lr: {args.lr * args.lr_backbone_multiplier}")
    log.info("----------------------------------------------------------\n")

    # --- REPLACED: PolyLR with OneCycleLR --- #
    log.info(f"--- Setting up OneCycleLR Scheduler ---")
    log.info(f"Max LR: {args.lr}, Epochs: {args.epochs}, Steps per Epoch: {len(train_dataLoader)}")
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in optimizer.param_groups], # Pass max_lr for each param group
        steps_per_epoch=len(train_dataLoader),
        epochs=args.epochs,
        pct_start=0.3, # Use 30% of steps for warm-up
        div_factor=25, # Initial LR is max_lr / 25
        final_div_factor=1e4 # Final LR is max_lr / 1e4
    )

    # 6. TRAINING LOOP
    log.info("--- Starting Training ---")
    start_time = time.time()
    best_mIoU = 0.0
    best_metrics_from_best_mIoU_epoch = {}
    log_data = []

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        log.info(f"\n===== Epoch {epoch}/{args.epochs - 1} =====")
        # --- UPDATED: Pass the scheduler to the training function --- #
        train_stats = train_one_epoch(model, criterion, train_dataLoader, optimizer, lr_scheduler, epoch, args, log)
        # --- REMOVED: lr_scheduler.step() is now called inside train_one_epoch --- #

        if device.type == 'cuda':
            log.info(f"GPU Memory: Allocated={torch.cuda.memory_allocated(0)/1024**2:.1f}MB, Peak={torch.cuda.max_memory_allocated(0)/1024**2:.1f}MB")
            torch.cuda.reset_peak_memory_stats(0)

        temp_eval_dir = output_dir / f'epoch_{epoch}_raw_preds'
        temp_eval_dir.mkdir(exist_ok=True)

        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        with torch.no_grad():
            model.eval()
            for data in tqdm(test_dl, desc=f"Testing Epoch {epoch}"):
                x, target = data["image"].to(device), data["label"].cpu().numpy()
                out = model(x).cpu().numpy()
                root_name = data["A_paths"][0].split("/")[-1][0:-4]
                cv2.imwrite(str(temp_eval_dir / f"{root_name}_pre.png"), (255 * out[0, 0]).astype(np.uint8))
                cv2.imwrite(str(temp_eval_dir / f"{root_name}_lab.png"), (255 * target[0, 0]).astype(np.uint8))

        current_epoch_metrics = eval(log, str(temp_eval_dir), epoch)
        
        # --- MODIFIED: Replaced with a formatted epoch summary --- #
        log.info(f"---------- Epoch {epoch} Summary ----------")
        log.info(f"  - Train Loss: {train_stats['loss']:.4f}")
        log.info(f"  - mIoU:       {current_epoch_metrics.get('mIoU', 0):.4f}")
        log.info(f"  - ODS:        {current_epoch_metrics.get('ODS', 0):.4f}")
        log.info(f"  - OIS:        {current_epoch_metrics.get('OIS', 0):.4f}")
        log.info(f"  - F1:         {current_epoch_metrics.get('F1', 0):.4f}")
        log.info(f"  - Precision:  {current_epoch_metrics.get('Precision', 0):.4f}")
        log.info(f"  - Recall:     {current_epoch_metrics.get('Recall', 0):.4f}")
        log.info(f"-----------------------------------")

        log_entry = {'epoch': epoch, 'train_loss': train_stats['loss'], **current_epoch_metrics}
        log_data.append(log_entry)

        utils.save_on_master({
            'model': model.state_dict(), 'epoch': epoch
        }, weights_dir / 'checkpoint_last.pth')

        # --- MODIFIED: Make robust to missing keys --- #
        if current_epoch_metrics.get('mIoU', 0) > best_mIoU:
            best_mIoU = current_epoch_metrics.get('mIoU', 0)
            best_metrics_from_best_mIoU_epoch = current_epoch_metrics
            log.info(f"*** New best mIoU: {best_mIoU:.4f} at epoch {epoch}! Saving best model and masks. ***")

            utils.save_on_master({'model': model.state_dict()}, weights_dir / 'checkpoint_best.pth')
            save_best_masks(model, device, args, masks_dir)

        log.info(f"Epoch {epoch} finished in {(time.time() - epoch_start_time):.2f}s. Current best mIoU: {best_mIoU:.4f}")

    # 7. FINALIZATION
    total_time = time.time() - start_time
    log.info(f"--- Training Finished in {datetime.timedelta(seconds=int(total_time))} ---")

    if best_metrics_from_best_mIoU_epoch:
        best_precision = best_metrics_from_best_mIoU_epoch.get('Precision', 0)
        best_recall = best_metrics_from_best_mIoU_epoch.get('Recall', 0)

        if (best_precision + best_recall) > 0:
            recalculated_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall)
        else:
            recalculated_f1 = 0.0

        final_report = {
            'Best_Epoch': best_metrics_from_best_mIoU_epoch.get('epoch'),
            'mIoU': best_metrics_from_best_mIoU_epoch.get('mIoU'),
            'ODS': best_metrics_from_best_mIoU_epoch.get('ODS'),
            'OIS': best_metrics_from_best_mIoU_epoch.get('OIS'),
            'Precision': best_precision,
            'Recall': best_recall,
            'F1_Score (Recalculated)': recalculated_f1
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
