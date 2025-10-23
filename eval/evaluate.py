
import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

def _get_statistics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    return tp, fp, fn

def calculate_metrics(pred_list: list, gt_list: list) -> dict:
    thresholds = np.arange(0.0, 1.0, 0.01)

    image_f1_scores = np.zeros((len(pred_list), len(thresholds)))
    image_miou_scores = np.zeros((len(pred_list), len(thresholds)))

    gt_masks_binary = [(gt > 127).astype(np.uint8) for gt in gt_list]

    for img_idx, (prob_map_uint8, gt_mask) in enumerate(zip(pred_list, gt_masks_binary)):
        prob_map_float = prob_map_uint8.astype(np.float32) / 255.0

        for thresh_idx, t in enumerate(thresholds):
            pred_mask = (prob_map_float > t).astype(np.uint8)
            tp, fp, fn = _get_statistics(pred_mask, gt_mask)

            if (tp + fn) == 0:
                current_f1 = 1.0 if (tp + fp) == 0 else 0.0
            else:
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                current_f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            image_f1_scores[img_idx, thresh_idx] = current_f1

            tn = np.sum((pred_mask == 0) & (gt_mask == 0))
            iou_foreground = tp / (tp + fp + fn + 1e-8)
            iou_background = tn / (tn + fp + fn + 1e-8)
            current_miou = (iou_foreground + iou_background) / 2
            image_miou_scores[img_idx, thresh_idx] = current_miou

    avg_f1_per_threshold = np.mean(image_f1_scores, axis=0)
    best_ods_f1_idx = np.argmax(avg_f1_per_threshold)
    ods_f1 = avg_f1_per_threshold[best_ods_f1_idx]
    best_threshold_for_ods_f1 = thresholds[best_ods_f1_idx]

    best_f1s_per_image = np.max(image_f1_scores, axis=1)
    ois_f1 = np.mean(best_f1s_per_image)

    avg_miou_per_threshold = np.mean(image_miou_scores, axis=0)
    best_ods_miou_idx = np.argmax(avg_miou_per_threshold)
    ods_miou = avg_miou_per_threshold[best_ods_miou_idx]

    global_tp, global_fp, global_fn = 0, 0, 0
    for i in range(len(pred_list)):
        prob_map = pred_list[i].astype(np.float32) / 255.0
        pred_mask = (prob_map > best_threshold_for_ods_f1).astype(np.uint8)
        tp, fp, fn = _get_statistics(pred_mask, gt_masks_binary[i])
        global_tp += tp
        global_fp += fp
        global_fn += fn

    ods_precision = global_tp / (global_tp + global_fp + 1e-8)
    ods_recall = global_tp / (global_tp + global_fn + 1e-8)
    fg_iou_at_ods = global_tp / (global_tp + global_fp + global_fn + 1e-8)

    return {
        'mIoU': ods_miou,
        'Foreground_IoU': fg_iou_at_ods,
        'ODS': ods_f1,
        'OIS': ois_f1,
        'F1': ods_f1,
        'Precision': ods_precision,
        'Recall': ods_recall,
        'best_threshold': int(best_threshold_for_ods_f1 * 255)
    }

def imread(path, load_mode=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(path, load_mode)
    if img is None:
        print(f"Warning: Failed to read image at {path}")
    return img

def get_image_pairs(data_dir, suffix_gt, suffix_pred):
    gt_paths = sorted(glob.glob(os.path.join(data_dir, f'*{suffix_gt}.png')))
    
    if not gt_paths:
        print(f"Warning: No ground truth files found in {data_dir} with suffix '{suffix_gt}.png'.")
        return [], []

    pred_list = []
    gt_list = []

    iterable = tqdm(gt_paths, desc="Loading image pairs") if len(gt_paths) > 100 else gt_paths

    for gt_path in iterable:
        pred_path = gt_path.replace(suffix_gt, suffix_pred)
        if not os.path.exists(pred_path):
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
    suffix_gt = "_lab"
    suffix_pred = "_pre"

    pred_list, gt_list = get_image_pairs(results_dir, suffix_gt, suffix_pred)

    if not pred_list or not gt_list:
        log_eval.warning("Evaluation skipped: No valid image pairs found.")
        return {'epoch': epoch, 'mIoU': 0, 'Foreground_IoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0, 'best_threshold': 0}

    metrics = calculate_metrics(pred_list, gt_list)
    metrics['epoch'] = epoch

    return metrics
