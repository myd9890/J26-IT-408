"""
Loss Functions for Semantic Segmentation.

Implements:
1. SoftDiceLoss: Multi-Class Soft Dice Loss with Laplace smoothing (epsilon=1e-6).
2. CombinedDiceCrossEntropyLoss: Weighted combination of Soft Dice + Cross-Entropy Loss
   (default: 0.5 Dice + 0.5 CE).
"""

from typing import Optional
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
