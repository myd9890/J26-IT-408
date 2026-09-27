"""
Loss Functions and Evaluation Metrics for Semantic Segmentation.

Implements:
1. CombinedDiceCrossEntropyLoss: Joint Multi-Class Soft Dice + Cross-Entropy Loss
   with configurable weighting (default: 0.5 Dice + 0.5 CE).
2. compute_segmentation_metrics: Computes IoU, Dice, Precision, Recall, and
   Pixel Accuracy for Leaf and Lesion classes, as well as Macro averages.
"""

from typing import Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Multi-Class Soft Dice Loss with Laplace smoothing (epsilon=1e-6).
    Computes smooth Dice similarity across target classes.
    """

    def __init__(self, epsilon: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) unnormalized predictions.
            target: (B, H, W) ground truth integer class labels.

        Returns:
            Scalar Dice loss = 1.0 - mean(Dice).
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Convert target to one-hot encoding: (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            p_c = probs[:, c, :, :]
            y_c = target_one_hot[:, c, :, :]

            intersection = torch.sum(p_c * y_c, dim=(1, 2))
            cardinality = torch.sum(p_c ** 2 + y_c ** 2, dim=(1, 2))

            dice = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)
            dice_per_class.append(dice.mean())

        mean_dice = torch.stack(dice_per_class).mean()
        return 1.0 - mean_dice


class CombinedDiceCrossEntropyLoss(nn.Module):
    """
    Weighted combination of Multi-Class Soft Dice Loss and Cross-Entropy Loss.
    Loss = dice_weight * DiceLoss + ce_weight * CELoss.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = SoftDiceLoss(epsilon=epsilon)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) model predictions.
            target: (B, H, W) ground truth classes in {0..C-1}.

        Returns:
            Combined scalar loss tensor.
        """
        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        return self.dice_weight * dice + self.ce_weight * ce


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
        iou = (tp + epsilon) / (union + epsilon) if union > 0 or (tp == 0 and fp == 0 and fn == 0) else 0.0
        # If both pred and target have zero pixels for this class, IoU is 1.0
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
