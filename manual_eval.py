# final_evaluate.py (Fixed OIS & ODS Calculation Logic)
# Author: Roy (with Gemini Assistant)
# Description: A standalone script to calculate segmentation metrics
#              from a directory containing both prediction (_pre.png)
#              and ground truth (_lab.png) files.

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse


# ----------------- 核心辅助函数 (保持不变) -----------------

def calculate_stats(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    """计算单个预测结果的基础统计数据 (TP, FP, FN, TN)。"""
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))
    return tp, fp, fn, tn


def f1_score_from_stats(tp: int, fp: int, fn: int) -> float:
    """根据统计数据计算 F1-score。"""
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def miou_from_stats(tp: int, fp: int, fn: int, tn: int) -> float:
    """根据统计数据计算 mIoU。"""
    epsilon = 1e-8
    iou_foreground = tp / (tp + fp + fn + epsilon)
    iou_background = tn / (tn + fp + fn + epsilon)
    miou = (iou_foreground + iou_background) / 2
    return miou


# ----------------- ODS, OIS 及其他指标的主计算函数 (修正后的逻辑) -----------------

def calculate_all_metrics(all_prob_maps: list, all_gt_masks: list) -> dict:
    """
    一次性高效计算 ODS, OIS, mIoU, Precision, 和 Recall。
    修正逻辑：
    - ODS: 在数据集级别找到一个全局阈值，使所有图片在该阈值下的F1分数的平均值最大
    - OIS: 对每张图片找到其最佳F1分数，然后对所有分数求平均
    """
    thresholds = np.arange(1, 256)

    # 存储每张图片在每个阈值下的F1分数
    image_f1_scores = np.zeros((len(all_prob_maps), len(thresholds)))
    # 存储每张图片在每个阈值下的mIoU分数
    image_miou_scores = np.zeros((len(all_prob_maps), len(thresholds)))

    print("Accumulating statistics across all thresholds and images...")
    for img_idx, (prob_map, gt_mask) in enumerate(tqdm(zip(all_prob_maps, all_gt_masks), total=len(all_prob_maps))):
        is_gt_empty = (np.sum(gt_mask) == 0)

        for thresh_idx, t_val in enumerate(thresholds):
            pred_mask = (prob_map >= t_val).astype(np.uint8)
            tp, fp, fn, tn = calculate_stats(pred_mask, gt_mask)

            # 特殊处理：如果真值图为空
            if is_gt_empty:
                # 如果预测也为空(tp+fp=0)，则F1为1 (完美匹配)，否则为0
                current_f1 = 1.0 if (tp + fp == 0) else 0.0
                current_miou = 1.0 if (tp + fp == 0) else 0.0  # 保守处理
            else:
                current_f1 = f1_score_from_stats(tp, fp, fn)
                current_miou = miou_from_stats(tp, fp, fn, tn)

            image_f1_scores[img_idx, thresh_idx] = current_f1
            image_miou_scores[img_idx, thresh_idx] = current_miou

    # --- ODS 计算 (修正：使用平均F1而非全局统计) ---
    print("Calculating ODS F1-Score...")
    # 对每个阈值，计算所有图片的F1平均值
    avg_f1_per_threshold = np.mean(image_f1_scores, axis=0)
    best_ods_idx = np.argmax(avg_f1_per_threshold)
    ods_f1 = avg_f1_per_threshold[best_ods_idx]

    # --- OIS 计算 (保持不变，正确) ---
    print("Calculating OIS F1-Score...")
    best_f1s_per_image = np.max(image_f1_scores, axis=1)
    ois_f1 = np.mean(best_f1s_per_image)

    # --- Max mIoU 计算 (类似OIS，但计算mIoU) ---
    print("Calculating Max mIoU (per-image optimal)...")
    best_miou_per_image = np.max(image_miou_scores, axis=1)
    max_miou = np.mean(best_miou_per_image)

    # --- 其他指标计算 (在ODS最佳阈值下) ---
    print("Calculating other metrics at ODS threshold...")
    best_threshold = thresholds[best_ods_idx]
    # 在ODS最佳阈值下，计算全局统计（用于Precision/Recall/mIoU）
    global_tp, global_fp, global_fn, global_tn = 0, 0, 0, 0
    for i in range(len(all_prob_maps)):
        pred_mask = (all_prob_maps[i] >= best_threshold).astype(np.uint8)
        tp, fp, fn, tn = calculate_stats(pred_mask, all_gt_masks[i])
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

    # 计算ODS阈值下的指标
    ods_precision = global_tp / (global_tp + global_fp + 1e-8)
    ods_recall = global_tp / (global_tp + global_fn + 1e-8)
    iou_foreground = global_tp / (global_tp + global_fp + global_fn + 1e-8)
    iou_background = global_tn / (global_tn + global_fp + global_fn + 1e-8)
    mIoU_at_ods = (iou_foreground + iou_background) / 2

    return {
        "ODS": ods_f1,
        "OIS": ois_f1,
        "mIoU_at_ODS": mIoU_at_ods,
        "Max_mIoU": max_miou,
        "Foreground_IoU": iou_foreground,
        "Precision": ods_precision,
        "Recall": ods_recall,
        "Best_Threshold": best_threshold
    }


# ----------------- 主执行函数 (已修改) -----------------

def main(args):
    """主函数，负责加载数据、调用计算并打印结果。"""

    eval_dir = args.eval_dir

    # 检查路径
    if not os.path.isdir(eval_dir):
        print(f"错误：评估目录不存在 -> {eval_dir}")
        return

    # --- 加载数据 ---
    print(f"正在从目录 '{eval_dir}' 中加载预测图和真值图标注...")
    all_prob_maps = []
    all_gt_masks = []

    # 找到所有的预测文件
    pred_files = sorted([f for f in os.listdir(eval_dir) if f.endswith('_pre.png')])

    if not pred_files:
        print(f"错误：在 '{eval_dir}' 中未找到任何 '_pre.png' 预测文件。")
        return

    for pred_filename in tqdm(pred_files, desc="Loading data"):
        # 构造对应的真值文件名
        gt_filename = pred_filename.replace('_pre.png', '_lab.png')

        pred_path = os.path.join(eval_dir, pred_filename)
        gt_path = os.path.join(eval_dir, gt_filename)

        # 确认真值文件存在
        if not os.path.exists(gt_path):
            print(f"警告：找不到对应的真值文件 {gt_path}，已跳过 {pred_filename}")
            continue

        # 加载 0-255 的灰度概率图
        prob_map = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # 加载真值图并二值化为 0/1
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_img > 127).astype(np.uint8)

        # 检查维度是否匹配
        if prob_map.shape != gt_mask.shape:
            print(f"警告：尺寸不匹配！跳过 {pred_filename}。预测图: {prob_map.shape}, 真值图: {gt_mask.shape}")
            continue

        all_prob_maps.append(prob_map)
        all_gt_masks.append(gt_mask)

    if not all_prob_maps:
        print("错误：未能加载任何有效的图像对。请检查文件命名和内容。")
        return

    # --- 计算所有指标 ---
    metrics = calculate_all_metrics(all_prob_maps, all_gt_masks)

    # --- 打印结果 ---
    print("\n" + "=" * 40)
    print(" " * 12 + "评 估 结 果")
    print("=" * 40)
    print(f"  - ODS (Optimal Dataset F1-Score): {metrics['ODS']:.4f}")
    print(f"  - OIS (Optimal Image F1-Score):   {metrics['OIS']:.4f}")
    print(f"  (Note: Theoretically, OIS should be >= ODS)")
    print("-" * 20)
    print(f"  - mIoU (at ODS threshold):        {metrics['mIoU_at_ODS']:.4f}")
    print(f"  - Max mIoU (per-image optimal):   {metrics['Max_mIoU']:.4f}")
    print(f"  - Foreground IoU (at ODS thresh): {metrics['Foreground_IoU']:.4f}")
    print("-" * 20)
    print(f"  Metrics at Best ODS Threshold (Value = {metrics['Best_Threshold']}):")
    print(f"    - Precision:                    {metrics['Precision']:.4f}")
    print(f"    - Recall:                       {metrics['Recall']:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    # --- 命令行参数解析器 ---
    parser = argparse.ArgumentParser(description="Calculate ODS, OIS, and other metrics for crack segmentation.")
    parser.add_argument('--eval_dir', type=str, required=True,
                        help="Directory containing both prediction probability maps (_pre.png) and ground truth label maps (_lab.png).")

    args = parser.parse_args()
    main(args)