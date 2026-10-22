"""
Segmentation modules for Banana Foliar Disease Assessment (FR2, FR3).
"""

from src.segmentation.classical_baseline import ClassicalSegmenter
from src.segmentation.unet import UNet
from src.segmentation.loss_and_metrics import (
    SoftDiceLoss,
    CombinedDiceCrossEntropyLoss,
    compute_segmentation_metrics,
)
from src.segmentation.scope_calculator import (
    MorphometricFeatures,
    extract_leaf_and_lesion_features,
)
from src.segmentation.train_seg import train_unet

__all__ = [
    "ClassicalSegmenter",
    "UNet",
    "SoftDiceLoss",
    "CombinedDiceCrossEntropyLoss",
    "compute_segmentation_metrics",
    "MorphometricFeatures",
    "extract_leaf_and_lesion_features",
    "train_unet",
]
