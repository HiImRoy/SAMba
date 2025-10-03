# eval/evaluate.py
# Author: Roy (with Gemini Assistant)
# **重构**: 本文件中的指标计算逻辑已根据 manual_eval.py 中的精确定义进行了完全重写，
# 以确保训练过程中的 ODS, OIS, 和 mIoU 指标与最终手动评估时完全一致。

import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

# --- 核心计算逻辑 (从 manual_eval.py 迁移并重构) ---

def _calculate_stats(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    """计算单个预测结果的基础统计数据 (TP, FP, FN, TN)。"""
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))
    return tp, fp, fn, tn

def _f1_score_from_stats(tp: int, fp: int, fn: int) -> float:
    """根据统计数据计算 F1-score。"""
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1

def _miou_from_stats(tp: int, fp: int, fn: int, tn: int) -> tuple:
    """根据统计数据计算前景IoU和mIoU。"""
    epsilon = 1e-8
    iou_foreground = tp / (tp + fp + fn + epsilon)
    iou_background = tn / (tn + fp + fn + epsilon)
    miou = (iou_foreground + iou_background) / 2
    return iou_foreground, miou

def calculate_metrics(pred_list: list, gt_list: list, num_cls: int = 2) -> dict:
    """ (重写后的函数)
    一次性高效计算 ODS, OIS, mIoU, Precision, 和 Recall。
    采用 manual_eval.py 中的精确计算逻辑。
    """
    thresholds = np.arange(1, 256)
    
    # 存储每张图片在每个阈值下的 F1 分数和 mIoU 分数
    image_f1_scores = np.zeros((len(pred_list), len(thresholds)))
    
    # 预处理真值图
    gt_masks_binary = [(gt / 255 > 0.5).astype(np.uint8) for gt in gt_list]

    # 1. 累积每个图像在每个阈值下的统计数据
    for img_idx, (prob_map, gt_mask) in enumerate(zip(pred_list, gt_masks_binary)):
        is_gt_empty = (np.sum(gt_mask) == 0)
        for thresh_idx, t_val in enumerate(thresholds):
            pred_mask = (prob_map >= t_val).astype(np.uint8)
            tp, fp, fn, _ = _calculate_stats(pred_mask, gt_mask)

            if is_gt_empty:
                current_f1 = 1.0 if (tp + fp == 0) else 0.0
            else:
                current_f1 = _f1_score_from_stats(tp, fp, fn)
            
            image_f1_scores[img_idx, thresh_idx] = current_f1

    # 2. 计算 ODS (Optimal Dataset Score)
    # 对每个阈值，计算所有图片的F1平均值，然后找到最佳平均值
    avg_f1_per_threshold = np.mean(image_f1_scores, axis=0)
    best_ods_idx = np.argmax(avg_f1_per_threshold)
    ods_f1 = avg_f1_per_threshold[best_ods_idx]
    best_threshold_for_ods = thresholds[best_ods_idx]

    # 3. 计算 OIS (Optimal Image Score)
    # 对每张图片找到其最佳F1分数，然后对所有分数求平均
    best_f1s_per_image = np.max(image_f1_scores, axis=1)
    ois_f1 = np.mean(best_f1s_per_image)

    # 4. 在ODS最佳阈值下，计算全局的 mIoU, Precision, Recall, Foreground_IoU
    global_tp, global_fp, global_fn, global_tn = 0, 0, 0, 0
    for i in range(len(pred_list)):
        pred_mask = (pred_list[i] >= best_threshold_for_ods).astype(np.uint8)
        tp, fp, fn, tn = _calculate_stats(pred_mask, gt_masks_binary[i])
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

    ods_precision = global_tp / (global_tp + global_fp + 1e-8)
    ods_recall = global_tp / (global_tp + global_fn + 1e-8)
    fg_iou, miou_at_ods = _miou_from_stats(global_tp, global_fp, global_fn, global_tn)

    # 5. 组装并返回与 main.py 兼容的指标字典
    return {
        'mIoU': miou_at_ods,          # 使用在ODS阈值下的mIoU作为主要mIoU指标
        'Foreground_IoU': fg_iou,     # 在ODS阈值下的前景IoU
        'ODS': ods_f1,               # ODS F1-Score
        'OIS': ois_f1,               # OIS F1-Score
        'F1': ods_f1,                 # 将 ODS F1 作为主要的 F1 指标
        'Precision': ods_precision,    # 在ODS阈值下的精度
        'Recall': ods_recall,         # 在ODS阈值下的召回率
        'best_threshold': int(best_threshold_for_ods) # ODS对应的最佳阈值
    }

# --- 数据加载与评估流程 (保持不变) ---

def imread(path, load_mode=cv2.IMREAD_GRAYSCALE):
    """Helper to read image."""
    return cv2.imread(path, load_mode)

def get_image_pairs(data_dir, suffix_gt, suffix_pred):
    """Loads prediction and ground truth images from a directory."""
    gt_paths = glob.glob(os.path.join(data_dir, f'*{suffix_gt}.png'))
    
    if not gt_paths:
        print(f"Warning: No ground truth files found in {data_dir} with suffix '{suffix_gt}.png'.")
        return [], []

    pred_paths = [p.replace(suffix_gt, suffix_pred) for p in gt_paths]
    
    pred_imgs = [imread(p) for p in tqdm(pred_paths, desc="Loading Predictions")]
    gt_imgs = [imread(p) for p in tqdm(gt_paths, desc="Loading Ground Truth")]

    loaded_pred_imgs = [img for img in pred_imgs if img is not None]
    loaded_gt_imgs = [gt_imgs[i] for i, img in enumerate(pred_imgs) if img is not None]

    if len(loaded_pred_imgs) != len(pred_paths):
        print(f"Warning: Some prediction images could not be loaded.")

    return loaded_pred_imgs, loaded_gt_imgs

def eval(log_eval, results_dir, epoch):
    """Main evaluation function called by main.py."""
    suffix_gt = "_lab"
    suffix_pred = "_pre"
    log_eval.info(f"Evaluating results in: {results_dir}")

    pred_list, gt_list = get_image_pairs(results_dir, suffix_gt, suffix_pred)

    if not pred_list or not gt_list:
        log_eval.warning("Evaluation skipped: No image pairs found.")
        return {'epoch': epoch, 'mIoU': 0, 'Foreground_IoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0, 'best_threshold': 127}

    # 调用重写后的核心计算函数
    metrics = calculate_metrics(pred_list, gt_list)
    metrics['epoch'] = epoch

    # 日志打印部分保持不变，会自动显示新计算出的指标
    log_eval.info(f"mIoU (at ODS) -> {metrics['mIoU']:.4f}")
    log_eval.info(f"Foreground_IoU (at ODS) -> {metrics.get('Foreground_IoU', 0.0):.4f}")
    log_eval.info(f"ODS -> {metrics['ODS']:.4f}")
    log_eval.info(f"OIS -> {metrics['OIS']:.4f}")
    log_eval.info(f"Precision (at ODS) -> {metrics['Precision']:.4f}")
    log_eval.info(f"Recall (at ODS) -> {metrics['Recall']:.4f}")
    log_eval.info(f"Best Threshold (for ODS) -> {metrics['best_threshold']}")
    log_eval.info("Evaluation finished!")

    return metrics
