"""
PyTorch 3-Class U-Net Semantic Segmentation Architecture.

(Compatibility Facade - logic moved to src.models.segmentation.unet)
"""

from src.models.segmentation.unet import (
    DoubleConv,
    Down,
    Up,
    OutConv,
    UNet,
)

__all__ = [
    "DoubleConv",
    "Down",
    "Up",
    "OutConv",
    "UNet",
]
