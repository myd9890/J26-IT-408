"""
Loss Functions and Evaluation Metrics for Semantic Segmentation.

(Compatibility Facade - logic moved to src.training.losses and src.evaluation.metrics)
"""

from src.training.losses import SoftDiceLoss, CombinedDiceCrossEntropyLoss
from src.evaluation.metrics import compute_segmentation_metrics

__all__ = [
    "SoftDiceLoss",
    "CombinedDiceCrossEntropyLoss",
    "compute_segmentation_metrics",
]
