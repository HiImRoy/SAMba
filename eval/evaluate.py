# eval/evaluate.py
# Author: Roy
# **重构**: 本文件中的指标计算逻辑已根据 manual_eval.py 中的精确定义进行了完全重写，
# 以确保训练过程中的 ODS, OIS, 和 mIoU 指标与最终手动评估时完全一致。

import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

# ----------------- 核心辅助函数 (与 manual_eval.py 一致) -----------------

def _get_statistics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    """计算 TP, FP, FN。"""
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    return tp, fp, fn

# ----------------- 核心指标计算 (完全对齐 manual_eval.py 并兼容 main.py) -----------------

def calculate_metrics(pred_list: list, gt_list: list) -> dict:
    """ (重写后的函数)
    一次性高效计算 ODS, OIS, Max_mIoU, Precision, 和 Recall。
    严格复现 manual_eval.py 中的精确计算逻辑，并额外计算 Foreground_IoU 以兼容 main.py。
    
    Args:
        pred_list (list): 0-255 范围的 uint8 概率图列表。
        gt_list (list): 0/255 范围的 uint8 真值图标注列表。
        
    Returns:
        dict: 包含所有计算指标的字典。
    """
    thresholds = np.arange(0.0, 1.0, 0.01)  # 使用 0.0-0.99 的浮点阈值

    # --- 数据结构初始化 ---
    image_f1_scores = np.zeros((len(pred_list), len(thresholds)))
    image_miou_scores = np.zeros((len(pred_list), len(thresholds)))

    # --- 累积每个图像在每个阈值下的 F1 和 mIoU ---
    for img_idx, (prob_map_uint8, gt_mask_uint8) in enumerate(zip(pred_list, gt_list)):
        
        # 预处理: 将输入转换为正确的格式和类型
        prob_map_float = prob_map_uint8.astype(np.float32) / 255.0
        gt_mask = (gt_mask_uint8 > 127).astype(np.uint8)
        is_gt_empty = (np.sum(gt_mask) == 0)

        for thresh_idx, t in enumerate(thresholds):
            pred_mask = (prob_map_float > t).astype(np.uint8)
            tp, fp, fn = _get_statistics(pred_mask, gt_mask)

            # --- F1 Score 计算 ---
            if is_gt_empty:
                current_f1 = 1.0 if (tp + fp == 0) else 0.0
            else:
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                current_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            image_f1_scores[img_idx, thresh_idx] = current_f1

            # --- mIoU Score 计算 ---
            tn = np.sum((pred_mask == 0) & (gt_mask == 0))
            iou_foreground = tp / (tp + fp + fn + 1e-8)
            iou_background = tn / (tn + fp + fn + 1e-8)
            current_miou = (iou_foreground + iou_background) / 2
            image_miou_scores[img_idx, thresh_idx] = current_miou

    # --- 指标后处理 ---

    # 1. ODS 计算 (宏平均F1的最大值)
    avg_f1_per_threshold = np.mean(image_f1_scores, axis=0)
    best_ods_idx = np.argmax(avg_f1_per_threshold)
    ods_f1 = avg_f1_per_threshold[best_ods_idx]
    best_threshold_for_ods = thresholds[best_ods_idx]

    # 2. OIS 计算 (单图最优F1的平均值)
    best_f1s_per_image = np.max(image_f1_scores, axis=1)
    ois_f1 = np.mean(best_f1s_per_image)

    # 3. Max mIoU 计算 (单图最优mIoU的平均值)
    best_mious_per_image = np.max(image_miou_scores, axis=1)
    max_miou = np.mean(best_mious_per_image)

    # 4. 在 ODS 最佳阈值下，计算全局的 Precision, Recall, 和 Foreground_IoU
    global_tp, global_fp, global_fn = 0, 0, 0
    for i in range(len(pred_list)):
        gt_mask = (gt_list[i] > 127).astype(np.uint8)
        prob_map = pred_list[i].astype(np.float32) / 255.0
        pred_mask = (prob_map > best_threshold_for_ods).astype(np.uint8)
        tp, fp, fn = _get_statistics(pred_mask, gt_mask)
        global_tp += tp
        global_fp += fp
        global_fn += fn

    ods_precision = global_tp / (global_tp + global_fp + 1e-8)
    ods_recall = global_tp / (global_tp + global_fn + 1e-8)
    # 【新增】计算在ODS阈值下的全局前景IoU
    fg_iou_at_ods = global_tp / (global_tp + global_fp + global_fn + 1e-8)

    # 5. 组装并返回与 main.py 兼容的指标字典
    return {
        'mIoU': max_miou,             # 主要 mIoU 指标使用 Max_mIoU
        'Max_mIoU': max_miou,          # 显式提供 Max_mIoU
        'Foreground_IoU': fg_iou_at_ods, # 【新增】在ODS阈值下的前景IoU
        'ODS': ods_f1,                 # ODS F1-Score
        'OIS': ois_f1,                 # OIS F1-Score
        'F1': ods_f1,                  # 将 ODS F1 作为主要的 F1 指标
        'Precision': ods_precision,    # 在ODS阈值下的精度
        'Recall': ods_recall,          # 在ODS阈值下的召回率
        'best_threshold': int(best_threshold_for_ods * 255) # ODS对应的最佳阈值 (0-255范围)
    }

# --- 数据加载与评估流程 (保持不变) ---

def imread(path, load_mode=cv2.IMREAD_GRAYSCALE):
    """Helper to read image."""
    img = cv2.imread(path, load_mode)
    if img is None:
        print(f"Warning: Failed to read image at {path}")
    return img

def get_image_pairs(data_dir, suffix_gt, suffix_pred):
    """Loads prediction and ground truth images from a directory."""
    gt_paths = sorted(glob.glob(os.path.join(data_dir, f'*{suffix_gt}.png')))
    
    if not gt_paths:
        print(f"Warning: No ground truth files found in {data_dir} with suffix '{suffix_gt}.png'.")
        return [], []

    pred_list = []
    gt_list = []

    for gt_path in tqdm(gt_paths, desc="Loading image pairs"):
        pred_path = gt_path.replace(suffix_gt, suffix_pred)
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found for {gt_path}, skipping.")
            continue
        
        pred_img = imread(pred_path)
        gt_img = imread(gt_path)

        if pred_img is not None and gt_img is not None:
            if pred_img.shape != gt_img.shape:
                print(f"Warning: Shape mismatch for {pred_path} and {gt_path}, skipping.")
                continue
            pred_list.append(pred_img)
            gt_list.append(gt_img)

    return pred_list, gt_list

def eval(log_eval, results_dir, epoch):
    """Main evaluation function called by main.py."""
    suffix_gt = "_lab"
    suffix_pred = "_pre"
    log_eval.info(f"Evaluating results in: {results_dir}")

    pred_list, gt_list = get_image_pairs(results_dir, suffix_gt, suffix_pred)

    if not pred_list or not gt_list:
        log_eval.warning("Evaluation skipped: No valid image pairs found.")
        # 返回一个符合结构但值为0的字典
        return {'epoch': epoch, 'mIoU': 0, 'Max_mIoU': 0, 'Foreground_IoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0, 'best_threshold': 0}

    # 调用重写后的核心计算函数
    metrics = calculate_metrics(pred_list, gt_list)
    metrics['epoch'] = epoch

    # 日志打印部分更新以反映新的指标
    log_eval.info(f"Epoch [{epoch}] Evaluation Results:")
    log_eval.info(f"  - ODS (Optimal Dataset F1):   {metrics['ODS']:.4f}")
    log_eval.info(f"  - OIS (Optimal Image F1):     {metrics['OIS']:.4f}")
    log_eval.info(f"  - Max mIoU (per-image opt):   {metrics['Max_mIoU']:.4f}")
    log_eval.info(f"  - Fg IoU (at ODS thresh):     {metrics['Foreground_IoU']:.4f}")
    log_eval.info(f"  - Best Threshold (for ODS):   {metrics['best_threshold']}")
    log_eval.info(f"  - Precision (at ODS thresh):  {metrics['Precision']:.4f}")
    log_eval.info(f"  - Recall (at ODS thresh):     {metrics['Recall']:.4f}")
    log_eval.info("="*40)

    return metrics
