"""
Segmentation Models Package.

Exports:
- UNet: 3-Class Deep Learning Semantic Segmentation Network
- ClassicalSegmenter: Rule-Based ExG + CIELAB/HSV Baseline
"""

from src.models.segmentation.unet import UNet, DoubleConv, Down, Up, OutConv
from src.models.segmentation.classical import ClassicalSegmenter

__all__ = [
    "UNet",
    "DoubleConv",
    "Down",
    "Up",
    "OutConv",
    "ClassicalSegmenter",
]
