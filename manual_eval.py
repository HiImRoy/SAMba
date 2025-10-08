# manual_eval.py (Final Version, aligned with the reference script)
# Author: Roy (with Gemini Assistant)
# Description: This script implements the precise evaluation logic from the
#              reference code, including the macro-average based ODS calculation.

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import glob


# ----------------- 核心辅助函数 (与参考脚本一致) -----------------

def get_statistics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    """计算 TP, FP, FN。"""
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    return tp, fp, fn


def calculate_all_metrics(all_prob_maps: list, all_gt_masks: list) -> dict:
    """
    一次性高效计算所有关键指标，严格复现参考脚本的逻辑。
    - ODS: 宏平均F1的最大值。
    - OIS: 单图最优F1的平均值。
    - Max mIoU: 单图最优mIoU的平均值。
    """
    thresholds = np.arange(0.0, 1.0, 0.01)  # 对应 0-0.99 的阈值

    # --- 数据结构初始化 ---
    image_f1_scores = np.zeros((len(all_prob_maps), len(thresholds)))
    image_miou_scores = np.zeros((len(all_prob_maps), len(thresholds)))

    print("Accumulating F1 and mIoU scores for each image at each threshold...")
    for img_idx, (prob_map_uint8, gt_mask_uint8) in enumerate(
            tqdm(zip(all_prob_maps, all_gt_masks), total=len(all_prob_maps))):

        prob_map_float = prob_map_uint8.astype(np.float32) / 255.0
        gt_mask = (gt_mask_uint8 > 127).astype(np.uint8)
        is_gt_empty = (np.sum(gt_mask) == 0)

        for thresh_idx, t in enumerate(thresholds):
            pred_mask = (prob_map_float > t).astype(np.uint8)
            tp, fp, fn = get_statistics(pred_mask, gt_mask)

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
    #    对应参考代码中的 cal_ODS_metrics
    avg_f1_per_threshold = np.mean(image_f1_scores, axis=0)
    best_ods_idx = np.argmax(avg_f1_per_threshold)
    ods_f1 = avg_f1_per_threshold[best_ods_idx]
    best_threshold_for_ods = thresholds[best_ods_idx]

    # 2. OIS 计算
    #    对应参考代码中的 cal_OIS_metrics
    best_f1s_per_image = np.max(image_f1_scores, axis=1)
    ois_f1 = np.mean(best_f1s_per_image)

    # 3. Max mIoU 计算
    #    对应参考代码中的 cal_mIoU_metrics
    best_mious_per_image = np.max(image_miou_scores, axis=1)
    max_miou = np.mean(best_mious_per_image)

    # 4. 在 ODS 最佳阈值下，计算全局的 Precision 和 Recall
    #    这需要重新遍历一次，累加全局统计数据
    global_tp, global_fp, global_fn = 0, 0, 0
    for i in range(len(all_prob_maps)):
        gt_mask = (all_gt_masks[i] > 127).astype(np.uint8)
        prob_map = all_prob_maps[i].astype(np.float32) / 255.0
        pred_mask = (prob_map > best_threshold_for_ods).astype(np.uint8)
        tp, fp, fn = get_statistics(pred_mask, gt_mask)
        global_tp += tp
        global_fp += fp
        global_fn += fn

    ods_precision = global_tp / (global_tp + global_fp + 1e-8)
    ods_recall = global_tp / (global_tp + global_fn + 1e-8)

    return {
        "ODS": ods_f1,
        "OIS": ois_f1,
        "Max_mIoU": max_miou,
        "Precision": ods_precision,
        "Recall": ods_recall,
        "Best_Threshold": best_threshold_for_ods * 255  # 转换为 0-255 范围
    }


# ----------------- 主执行函数 -----------------

def main(args):
    """主函数，负责加载数据、调用计算并打印结果。"""

    eval_dir = args.eval_dir

    if not os.path.isdir(eval_dir):
        print(f"错误：评估目录不存在 -> {eval_dir}")
        return

    print(f"正在从目录 '{eval_dir}' 中加载预测图和真值图标注...")
    all_prob_maps = []
    all_gt_masks = []

    pred_files = sorted([f for f in os.listdir(eval_dir) if f.endswith('_pre.png')])

    if not pred_files:
        print(f"错误：在 '{eval_dir}' 中未找到任何 '_pre.png' 预测文件。")
        return

    for pred_filename in tqdm(pred_files, desc="Loading data"):
        gt_filename = pred_filename.replace('_pre.png', '_lab.png')
        pred_path = os.path.join(eval_dir, pred_filename)
        gt_path = os.path.join(eval_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"警告：找不到对应的真值文件 {gt_path}，已跳过 {pred_filename}")
            continue

        # 加载 0-255 的灰度概率图
        prob_map = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # 加载 0/255 的真值图
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if prob_map.shape != gt_img.shape:
            print(f"警告：尺寸不匹配！跳过 {pred_filename}。")
            continue

        all_prob_maps.append(prob_map)
        all_gt_masks.append(gt_img)

    if not all_prob_maps:
        print("错误：未能加载任何有效的图像对。")
        return

    metrics = calculate_all_metrics(all_prob_maps, all_gt_masks)

    print("\n" + "=" * 45)
    print(" " * 14 + "评 估 结 果 汇 总")
    print("=" * 45)
    print(f"  - ODS (Optimal Dataset F1-Score):   {metrics['ODS']:.4f}")
    print(f"  - OIS (Optimal Image F1-Score):     {metrics['OIS']:.4f}")
    print(f"  - Max mIoU (per-image optimal):     {metrics['Max_mIoU']:.4f}")
    print("-" * 45)
    print(f"  Metrics at Best ODS Threshold (Value = {metrics['Best_Threshold']:.0f}):")
    print(f"    - Precision:                      {metrics['Precision']:.4f}")
    print(f"    - Recall:                         {metrics['Recall']:.4f}")
    print("=" * 45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for crack segmentation, based on BSDS benchmark logic.")
    parser.add_argument('--eval_dir', type=str, required=True,
                        help="Directory containing both _pre.png (probability maps) and _lab.png (labels).")

    args = parser.parse_args()
    main(args)