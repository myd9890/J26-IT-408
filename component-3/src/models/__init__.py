"""
Models Root Package.

Exports:
- UNet, ClassicalSegmenter
- XGBoostSeverityEngine, fusion formulas
- SeverityGRU, calculate_ols_trend
"""

from src.models.segmentation import UNet, ClassicalSegmenter
from src.models.severity import (
    XGBoostSeverityEngine,
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
    get_recommended_action,
)
from src.models.temporal import SeverityGRU, calculate_ols_trend

__all__ = [
    "UNet",
    "ClassicalSegmenter",
    "XGBoostSeverityEngine",
    "fusion_multiplicative",
    "fusion_weighted_sum",
    "fusion_damped_hybrid",
    "map_score_to_category",
    "get_recommended_action",
    "SeverityGRU",
    "calculate_ols_trend",
]
