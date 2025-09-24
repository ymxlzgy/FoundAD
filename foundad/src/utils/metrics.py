import numpy as np
from sklearn import metrics
from skimage.measure import label, regionprops


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights,
    anomaly_ground_truth_labels
):
    # --- (1) Compute ROC-related metrics ---
    fpr, tpr, roc_thresholds = metrics.roc_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)

    # --- (2) Compute PR-related metrics ---
    precision, recall, pr_thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels,
                                                                      anomaly_prediction_weights)
    aupr = metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)

    # --- (3) Compute the maximum F1 score and its threshold ---
    # F1 = 2 * P * R / (P + R)
    F1_scores = 2.0 * precision * recall / np.clip(precision + recall, 1e-8, None)
    best_idx = np.argmax(F1_scores)
    
    # thresholds array is one element shorter than precision/recall,
    # so if best_idx == len(pr_thresholds), we handle it separately
    if best_idx < len(pr_thresholds):
        f1_max_threshold = pr_thresholds[best_idx]
    else:
        f1_max_threshold = 1.0  # or any default value
    f1_max = F1_scores[best_idx]

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "aupr": aupr,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "f1_max": f1_max,
        "f1_max_threshold": f1_max_threshold
    }


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations,
    ground_truth_masks
):
    # If input is a list, convert it to a NumPy array
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations, axis=0)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks, axis=0)

    # Flatten the arrays so we can feed them into scikit-learn metrics
    flat_scores = anomaly_segmentations.ravel().astype(np.float32)
    flat_labels = ground_truth_masks.ravel().astype(np.int32)

    # --- (1) Compute ROC-related metrics ---
    fpr, tpr, roc_thresholds = metrics.roc_curve(flat_labels, flat_scores)
    auroc = metrics.roc_auc_score(flat_labels, flat_scores)

    # --- (2) Compute PR-related metrics ---
    precision, recall, pr_thresholds = metrics.precision_recall_curve(flat_labels, flat_scores)
    aupr = metrics.average_precision_score(flat_labels, flat_scores)

    # --- (3) Compute the maximum F1 score and its threshold ---
    F1_scores = 2.0 * precision * recall / np.clip(precision + recall, 1e-8, None)
    best_idx = np.argmax(F1_scores)
    
    # Handle the index offset in thresholds
    if best_idx < len(pr_thresholds):
        f1_max_threshold = pr_thresholds[best_idx]
    else:
        f1_max_threshold = 1.0
    f1_max = F1_scores[best_idx]

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "aupr": aupr,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "f1_max": f1_max,
        "f1_max_threshold": f1_max_threshold
    }


def calculate_pro(masks, scores, max_steps=200, expect_fpr=0.3):
    thresholds = np.linspace(scores.min(), scores.max(), max_steps)
    pros = []
    fprs = []

    for threshold in thresholds:
        binary_scores = (scores > threshold).astype(int)

        # Calculate Pro
        pro_values = []
        for binary_score, mask in zip(binary_scores, masks):
            regions = regionprops(label(mask))
            for region in regions:
                tp_pixels = binary_score[region.coords[:, 0], region.coords[:, 1]].sum()
                pro_values.append(tp_pixels / region.area)
        pros.append(np.mean(pro_values))

        # Calculate FPR
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_scores).sum()
        fpr = fp_pixels / inverse_masks.sum()
        fprs.append(fpr)

    pros = np.array(pros)
    fprs = np.array(fprs)

    # Filter FPRs below the expected threshold
    valid_idxs = fprs <= expect_fpr
    fprs = fprs[valid_idxs]
    pros = pros[valid_idxs]

    # Normalize FPRs for AUC computation
    if len(fprs) > 1:
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = metrics.auc(fprs, pros) if len(fprs) > 1 else 0.0

    return pro_auc
