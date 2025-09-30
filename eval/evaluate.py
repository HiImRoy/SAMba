import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

def calculate_metrics(pred_list, gt_list, num_cls=2):
    """
    A unified and efficient function to calculate segmentation metrics including
    mIoU, ODS, OIS, and the associated Precision, Recall, and F1-score.

    This function iterates through thresholds only once to calculate all metrics.
    """
    # Statistics for ODS (per threshold, over all images)
    # Each element is [TP, FP, FN] for a given threshold
    stats_ods = [[0, 0, 0] for _ in range(256)]

    # Statistics for OIS (per image, per threshold)
    # Shape: (num_images, num_thresholds, 3) where 3 is for [TP, FP, FN]
    stats_ois = np.zeros((len(pred_list), 256, 3), dtype=np.int64)

    # Statistics for mIoU (per threshold, over all images)
    # Each element is [TP, FP, FN, TN] for a given threshold
    stats_miou = [[0, 0, 0, 0] for _ in range(256)]

    # Pre-convert ground truth images to binary
    gt_binary_list = [(gt / 255).astype('uint8') for gt in gt_list]

    # Iterate over each image pair
    for i, (pred_prob, gt_binary) in enumerate(zip(pred_list, gt_binary_list)):
        # Iterate over each possible threshold (0-255)
        for thresh in range(256):
            # Threshold the probability map to get a binary prediction
            pred_binary = (pred_prob > thresh).astype('uint8')

            # Calculate TP, FP, FN, TN for the current image at the current threshold
            tp = np.sum((pred_binary == 1) & (gt_binary == 1))
            fp = np.sum((pred_binary == 1) & (gt_binary == 0))
            fn = np.sum((pred_binary == 0) & (gt_binary == 1))
            tn = np.sum((pred_binary == 0) & (gt_binary == 0))

            # Accumulate stats for ODS
            stats_ods[thresh][0] += tp
            stats_ods[thresh][1] += fp
            stats_ods[thresh][2] += fn

            # Store stats for OIS
            stats_ois[i, thresh, 0] = tp
            stats_ois[i, thresh, 1] = fp
            stats_ois[i, thresh, 2] = fn

            # Accumulate stats for mIoU
            stats_miou[thresh][0] += tp
            stats_miou[thresh][1] += fp
            stats_miou[thresh][2] += fn
            stats_miou[thresh][3] += tn

    # --- Post-computation for ODS, Precision, Recall ---
    p_ods, r_ods, f1_ods = [], [], []
    for tp, fp, fn in stats_ods:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        p_ods.append(p)
        r_ods.append(r)
        f1_ods.append(f1)

    # Find the best threshold for ODS
    best_thresh_idx_ods = np.argmax(f1_ods)
    ods_f1 = f1_ods[best_thresh_idx_ods]
    ods_precision = p_ods[best_thresh_idx_ods]
    ods_recall = r_ods[best_thresh_idx_ods]

    # --- Post-computation for OIS ---
    f1_ois_per_image = []
    for i in range(len(pred_list)):
        tp = stats_ois[i, :, 0]
        fp = stats_ois[i, :, 1]
        fn = stats_ois[i, :, 2]
        p_ois = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        r_ois = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1_ois = np.divide(2 * p_ois * r_ois, p_ois + r_ois, out=np.zeros_like(p_ois, dtype=float), where=(p_ois + r_ois) != 0)
        f1_ois_per_image.append(np.max(f1_ois))
    ois_f1 = np.mean(f1_ois_per_image)

    # --- Post-computation for mIoU ---
    miou_list = []
    for tp, fp, fn, tn in stats_miou:
        iou_1 = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        iou_0 = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0
        miou_list.append((iou_1 + iou_0) / 2)
    best_miou = np.max(miou_list)

    # --- MODIFIED: Return the best threshold as well ---
    return {
        'mIoU': best_miou,
        'ODS': ods_f1,          # This is the main F1-score
        'OIS': ois_f1,
        'F1': ods_f1,           # Report ODS F1 as the primary F1
        'Precision': ods_precision, # Precision at the best ODS threshold
        'Recall': ods_recall,       # Recall at the best ODS threshold
        'best_threshold': best_thresh_idx_ods # The optimal threshold (0-255)
    }

def imread(path, load_mode=cv2.IMREAD_GRAYSCALE):
    """Helper to read image."""
    return cv2.imread(path, load_mode)

def get_image_pairs(data_dir, suffix_gt, suffix_pred):
    """Loads prediction and ground truth images from a directory."""
    # Use os.path.join for robust path construction
    gt_paths = glob.glob(os.path.join(data_dir, f'*{suffix_gt}.png'))
    
    if not gt_paths:
        print(f"Warning: No ground truth files found in {data_dir} with suffix '{suffix_gt}.png'.")
        return [], []

    pred_paths = [p.replace(suffix_gt, suffix_pred) for p in gt_paths]
    
    pred_imgs = [imread(p) for p in tqdm(pred_paths, desc="Loading Predictions")]
    gt_imgs = [imread(p) for p in tqdm(gt_paths, desc="Loading Ground Truth")]

    # Filter out pairs where an image failed to load
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

    # Load raw probability maps (as uint8) and ground truth maps
    pred_list, gt_list = get_image_pairs(results_dir, suffix_gt, suffix_pred)

    if not pred_list or not gt_list:
        log_eval.warning("Evaluation skipped: No image pairs found.")
        return {'epoch': epoch, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0, 'best_threshold': 127}

    # Calculate all metrics in one go
    metrics = calculate_metrics(pred_list, gt_list)
    metrics['epoch'] = epoch

    log_eval.info(f"mIoU -> {metrics['mIoU']:.4f}")
    log_eval.info(f"ODS -> {metrics['ODS']:.4f}")
    log_eval.info(f"OIS -> {metrics['OIS']:.4f}")
    log_eval.info(f"Precision (at ODS) -> {metrics['Precision']:.4f}")
    log_eval.info(f"Recall (at ODS) -> {metrics['Recall']:.4f}")
    log_eval.info(f"Best Threshold (for ODS) -> {metrics['best_threshold']}")
    log_eval.info("Evaluation finished!")

    return metrics
