import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score

def calculate_metrics_for_dir(pred_dir, gt_dir):
    """
    计算指定目录中所有预测图像的 F1 分数和 mIoU。

    Args:
        pred_dir (str): 预测结果图像（_pre.png）所在的文件夹路径。
        gt_dir (str): 真实标签图像（_lab.png）所在的文件夹路径。

    Returns:
        tuple: (平均F1分数, 平均IoU)
    """
    # 获取所有预测文件的路径
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_pre.png')]

    if not pred_files:
        print(f"错误：在目录 '{pred_dir}' 中未找到任何 '_pre.png' 预测文件。")
        return 0.0, 0.0

    all_f1s = []
    all_ious = []

    print(f"\n开始在 {len(pred_files)} 张图像上计算评估指标...")

    # 使用tqdm创建进度条
    for pred_filename in tqdm(pred_files, desc="正在评估"):
        # 从预测文件名构造真实标签文件名
        # 例如 'image1_pre.png' -> 'image1_lab.png'
        gt_filename = pred_filename.replace('_pre.png', '_lab.png')

        pred_path = os.path.join(pred_dir, pred_filename)
        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"警告：找不到对应的真实标签文件 {gt_path}，已跳过 {pred_filename}")
            continue

        # 以灰度模式加载图像
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # 1. 预处理：将图像转换为 0 和 1 的二值数组
        #    这是计算指标的标准做法
        pred_binary = (pred_img > 127).astype(np.uint8)
        gt_binary = (gt_img > 127).astype(np.uint8)

        # 2. 扁平化：将2D图像数组转换为1D向量，以供sklearn使用
        pred_flat = pred_binary.flatten()
        gt_flat = gt_binary.flatten()

        # 3. 计算当前图像的指标
        #    'binary'参数适用于二分类问题
        f1 = f1_score(gt_flat, pred_flat, average='binary', zero_division=0)
        iou = jaccard_score(gt_flat, pred_flat, average='binary', zero_division=0)

        all_f1s.append(f1)
        all_ious.append(iou)

    # 4. 计算所有图像的平均指标
    mean_f1 = np.mean(all_f1s) if all_f1s else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    return mean_f1, mean_iou

if __name__ == '__main__':
    # --- 配置您的路径 ---
    # 预测结果所在的文件夹 (test.py的输出目录)
    prediction_directory = './results/new_TUT_predictions'

    # 真实标签所在的文件夹 (这是您数据集中原始的测试集标签)
    # 注意：我们使用预测文件夹中的lab文件进行对比，因为它们是 test.py 直接生成的，确保了一一对应
    ground_truth_directory = './results/new_TUT_predictions'
    # --- 路径配置结束 ---

    # 检查路径是否存在
    if not os.path.isdir(prediction_directory):
        print(f"错误：预测目录不存在 -> {prediction_directory}")
        print("请先运行 test.py 生成预测结果。")
    elif not os.path.isdir(ground_truth_directory):
        print(f"错误：真实标签目录不存在 -> {ground_truth_directory}")
    else:
        # 计算并打印结果
        avg_f1, avg_miou = calculate_metrics_for_dir(prediction_directory, ground_truth_directory)

        print("\n" + "="*30)
        print("      评 估 结 果")
        print("="*30)
        print(f"平均 F1-Score: {avg_f1:.4f}")
        print(f"平均 mIoU    : {avg_miou:.4f}")
        print("="*30)