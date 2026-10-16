"""
Evaluation Metrics for Semantic Segmentation and ML Models.

Implements:
- compute_segmentation_metrics: Computes IoU, Dice, Precision, Recall, and
  Pixel Accuracy for Leaf and Lesion classes, as well as Macro averages.
"""

from typing import Dict, Union, Optional
import numpy as np
import torch


def compute_segmentation_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    num_classes: int = 3,
    epsilon: float = 1e-7
) -> Dict[str, float]:
    """
    Computes detailed semantic segmentation metrics across classes.

    Classes:
      0: Background
      1: Healthy leaf tissue
      2: Foliar lesion tissue

    Metrics returned:
      - overall_accuracy: Global pixel accuracy.
      - lesion_iou, lesion_dice, lesion_precision, lesion_recall: Metrics for lesion class (class 2).
      - leaf_iou, leaf_dice: Metrics for healthy leaf class (class 1).
      - total_leaf_iou, total_leaf_dice: Metrics for whole leaf lamina (classes 1 and 2 combined).
      - macro_iou, macro_dice: Arithmetic mean across all 3 classes.

    Args:
        pred: Predicted class mask tensor or numpy array of shape (H, W) or (B, H, W).
        target: Ground truth mask tensor or numpy array with same shape as pred.
        num_classes: Total number of classes (default 3).
        epsilon: Small numerical constant to avoid division by zero.

    Returns:
        Dictionary of float metrics rounded to 4 decimal places.
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
    else:
        pred_np = np.asarray(pred, dtype=np.int64)

    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().astype(np.int64)
    else:
        target_np = np.asarray(target, dtype=np.int64)

    if pred_np.shape != target_np.shape:
        raise ValueError(f"Shape mismatch: pred {pred_np.shape} vs target {target_np.shape}")

    # Overall pixel accuracy
    total_pixels = pred_np.size
    correct_pixels = np.count_nonzero(pred_np == target_np)
    overall_accuracy = float(correct_pixels / max(1, total_pixels))

    metrics: Dict[str, float] = {
        "overall_accuracy": round(overall_accuracy, 4)
    }

    class_ious = []
    class_dices = []

    for c in range(num_classes):
        pred_c = (pred_np == c)
        target_c = (target_np == c)

        tp = np.count_nonzero(pred_c & target_c)
        fp = np.count_nonzero(pred_c & ~target_c)
        fn = np.count_nonzero(~pred_c & target_c)

        union = tp + fp + fn
        if union == 0:
            iou = 1.0
            dice = 1.0
            prec = 1.0
            rec = 1.0
        else:
            iou = tp / (union + epsilon)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + epsilon)
            prec = tp / (tp + fp + epsilon)
            rec = tp / (tp + fn + epsilon)

        class_ious.append(iou)
        class_dices.append(dice)

        if c == 1:
            metrics["leaf_iou"] = round(float(iou), 4)
            metrics["leaf_dice"] = round(float(dice), 4)
        elif c == 2:
            metrics["lesion_iou"] = round(float(iou), 4)
            metrics["lesion_dice"] = round(float(dice), 4)
            metrics["lesion_precision"] = round(float(prec), 4)
            metrics["lesion_recall"] = round(float(rec), 4)

    # Whole leaf lamina metrics: (class 1 | class 2)
    pred_lamina = (pred_np == 1) | (pred_np == 2)
    target_lamina = (target_np == 1) | (target_np == 2)

    tp_lam = np.count_nonzero(pred_lamina & target_lamina)
    fp_lam = np.count_nonzero(pred_lamina & ~target_lamina)
    fn_lam = np.count_nonzero(~pred_lamina & target_lamina)
    union_lam = tp_lam + fp_lam + fn_lam

    if union_lam == 0:
        metrics["total_leaf_iou"] = 1.0
        metrics["total_leaf_dice"] = 1.0
    else:
        metrics["total_leaf_iou"] = round(float(tp_lam / (union_lam + epsilon)), 4)
        metrics["total_leaf_dice"] = round(float((2.0 * tp_lam) / (2.0 * tp_lam + fp_lam + fn_lam + epsilon)), 4)

    metrics["macro_iou"] = round(float(np.mean(class_ious)), 4)
    metrics["macro_dice"] = round(float(np.mean(class_dices)), 4)

    return metrics
