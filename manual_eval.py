# manual_eval.py (Strictly Aligned with Golden Standard + ODS Foreground IoU)
# Author: Roy (with Gemini Assistant)
# Description: This script implements evaluation logic strictly following
#              the golden standard, now including ODS Foreground IoU.

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


# ----------------- 核心辅助函数 (保持不变) -----------------

def get_statistics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    """计算 TP, FP, FN。"""
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    return tp, fp, fn


def calculate_all_metrics(all_prob_maps: list, all_gt_masks: list) -> dict:
    """
    一次性高效计算所有关键指标，计算逻辑严格遵循黄金基准。
    - ODS_F1: 数据集最优尺度的F1。
    - OIS_F1: 图像最优尺度的F1。
    - ODS_mIoU: 数据集最优尺度的mIoU。
    - ODS_Foreground_IoU: 数据集最优尺度的前景IoU (新增)。
    """
    thresholds = np.arange(0.0, 1.0, 0.01)

    # --- 数据结构初始化 ---
    image_f1_scores = np.zeros((len(all_prob_maps), len(thresholds)))
    image_miou_scores = np.zeros((len(all_prob_maps), len(thresholds)))
    # --- 【新增】为前景IoU初始化数据结构 ---
    image_fg_iou_scores = np.zeros((len(all_prob_maps), len(thresholds)))

    print("Accumulating F1, mIoU, and Foreground IoU scores for each image at each threshold...")
    gt_masks_binary = [(gt > 127).astype(np.uint8) for gt in all_gt_masks]

    for img_idx, (prob_map_uint8, gt_mask) in enumerate(
            tqdm(zip(all_prob_maps, gt_masks_binary), total=len(all_prob_maps))):

        prob_map_float = prob_map_uint8.astype(np.float32) / 255.0

        for thresh_idx, t in enumerate(thresholds):
            pred_mask = (prob_map_float > t).astype(np.uint8)
            tp, fp, fn = get_statistics(pred_mask, gt_mask)

            # --- F1 Score 计算 (已与 evaluate.py 对齐) ---
            precision = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + 1e-8)
            if (tp + fn) == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn + 1e-8)

            if (precision + recall) == 0:
                current_f1 = 0.0
            else:
                current_f1 = 2 * (precision * recall) / (precision + recall)
            image_f1_scores[img_idx, thresh_idx] = current_f1

            # --- mIoU Score 计算 (已与 evaluate.py 对齐) ---
            tn = np.sum((pred_mask == 0) & (gt_mask == 0))
            if (tp + fp + fn) == 0:
                current_miou = 0.0
            else:
                iou_foreground = tp / (tp + fp + fn + 1e-8)
                iou_background = tn / (tn + fp + fn + 1e-8)
                current_miou = (iou_foreground + iou_background) / 2
            image_miou_scores[img_idx, thresh_idx] = current_miou

            # --- 【新增】前景IoU (Foreground IoU) 计算 ---
            # 逻辑与 mIoU 中的 iou_foreground 部分一致
            if (tp + fp + fn) == 0:
                current_fg_iou = 0.0  # 遵循黄金标准对空前景并集的处理
            else:
                current_fg_iou = tp / (tp + fp + fn + 1e-8)
            image_fg_iou_scores[img_idx, thresh_idx] = current_fg_iou
            # --- 前景IoU 计算结束 ---

    # --- 指标后处理 ---

    # 1. ODS F1
    avg_f1_per_threshold = np.mean(image_f1_scores, axis=0)
    best_ods_f1_idx = np.argmax(avg_f1_per_threshold)
    ods_f1 = avg_f1_per_threshold[best_ods_f1_idx]
    best_threshold_for_ods_f1 = thresholds[best_ods_f1_idx]

    # 2. OIS F1
    best_f1s_per_image = np.max(image_f1_scores, axis=1)
    ois_f1 = np.mean(best_f1s_per_image)

    # 3. ODS mIoU
    avg_miou_per_threshold = np.mean(image_miou_scores, axis=0)
    best_ods_miou_idx = np.argmax(avg_miou_per_threshold)
    ods_miou = avg_miou_per_threshold[best_ods_miou_idx]
    best_threshold_for_ods_miou = thresholds[best_ods_miou_idx]

    # 4. 【新增】ODS Foreground IoU
    avg_fg_iou_per_threshold = np.mean(image_fg_iou_scores, axis=0)
    best_ods_fg_iou_idx = np.argmax(avg_fg_iou_per_threshold)
    ods_fg_iou = avg_fg_iou_per_threshold[best_ods_fg_iou_idx]
    best_threshold_for_ods_fg_iou = thresholds[best_ods_fg_iou_idx]

    # 在 ODS F1 最佳阈值下计算全局 P 和 R (辅助信息)
    global_tp, global_fp, global_fn = 0, 0, 0
    for i in range(len(all_prob_maps)):
        prob_map = all_prob_maps[i].astype(np.float32) / 255.0
        pred_mask = (prob_map > best_threshold_for_ods_f1).astype(np.uint8)
        tp, fp, fn = get_statistics(pred_mask, gt_masks_binary[i])
        global_tp += tp
        global_fp += fp
        global_fn += fn

    ods_precision = global_tp / (global_tp + global_fp + 1e-8)
    ods_recall = global_tp / (global_tp + global_fn + 1e-8)

    return {
        "ODS_F1": ods_f1,
        "OIS_F1": ois_f1,
        "ODS_mIoU": ods_miou,
        "ODS_Foreground_IoU": ods_fg_iou,  # 新增
        "Precision_at_ODS_F1": ods_precision,
        "Recall_at_ODS_F1": ods_recall,
        "Best_Threshold_ODS_F1": int(best_threshold_for_ods_f1 * 255),
        "Best_Threshold_ODS_mIoU": int(best_threshold_for_ods_miou * 255),
        "Best_Threshold_ODS_Foreground_IoU": int(best_threshold_for_ods_fg_iou * 255)  # 新增
    }


# ----------------- 主执行函数 (更新打印部分) -----------------
def main(args):
    """主函数，负责加载数据、调用计算并打印结果。"""
    eval_dir = args.eval_dir
    if not os.path.isdir(eval_dir):
        print(f"错误：评估目录不存在 -> {eval_dir}")
        return

    print(f"正在从目录 '{eval_dir}' 中加载预测图和真值图标注...")
    all_prob_maps = []
    all_gt_masks = []
    # 兼容 _pre.png 和 _pred.png
    pred_files = sorted(
        [f for f in os.listdir(eval_dir) if f.endswith('_pre.png')] +
        [f for f in os.listdir(eval_dir) if f.endswith('_pred.png')]
    )

    if not pred_files:
        print(f"错误：在 '{eval_dir}' 中未找到任何 '_pre.png' 或 '_pred.png' 预测文件。")
        return

    for pred_filename in tqdm(pred_files, desc="Loading data"):
        # 命名兼容
        gt_filename = pred_filename.replace('_pre.png', '_lab.png').replace('_pred.png', '_lab.png')
        pred_path = os.path.join(eval_dir, pred_filename)
        gt_path = os.path.join(eval_dir, gt_filename)

        if not os.path.exists(gt_path):
            gt_filename_alt = pred_filename.replace('_pre.png', '_gt.png').replace('_pred.png', '_gt.png')
            gt_path = os.path.join(eval_dir, gt_filename_alt)
            if not os.path.exists(gt_path):
                print(f"警告：找不到对应的真值文件 {gt_filename} 或 {gt_filename_alt}，已跳过 {pred_filename}")
                continue

        prob_map = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if prob_map is None or gt_img is None or prob_map.shape != gt_img.shape:
            print(f"警告：图像加载失败或尺寸不匹配！跳过 {pred_filename}。")
            continue

        all_prob_maps.append(prob_map)
        all_gt_masks.append(gt_img)

    if not all_prob_maps:
        print("错误：未能加载任何有效的图像对。")
        return

    metrics = calculate_all_metrics(all_prob_maps, all_gt_masks)

    print("\n" + "=" * 70)
    print(" " * 22 + "评 估 结 果 (黄金基准对齐)")
    print("=" * 70)
    print("--- F1-Score Based Metrics ---")
    print(f"  - ODS F1-Score:    {metrics['ODS_F1']:.4f} (at Thresh {metrics['Best_Threshold_ODS_F1']})")
    print(f"  - OIS F1-Score:    {metrics['OIS_F1']:.4f}")
    print("\n--- IoU Based Metrics ---")
    print(f"  - ODS mIoU:        {metrics['ODS_mIoU']:.4f} (at Thresh {metrics['Best_Threshold_ODS_mIoU']})")
    print(
        f"  - ODS Foreground IoU: {metrics['ODS_Foreground_IoU']:.4f} (at Thresh {metrics['Best_Threshold_ODS_Foreground_IoU']})")
    print("\n--- Metrics at ODS F1 Threshold ---")
    print(f"  - Precision:       {metrics['Precision_at_ODS_F1']:.4f}")
    print(f"  - Recall:          {metrics['Recall_at_ODS_F1']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Golden-standard aligned evaluation script for segmentation.")
    parser.add_argument('--eval_dir', type=str, required=True,
                        help="Directory containing prediction maps (_pre.png/_pred.png) and ground truth (_lab.png/_gt.png).")
    args = parser.parse_args()
    main(args)