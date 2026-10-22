"""
Severity Models Package.

Exports:
- fusion_rules: fusion_multiplicative, fusion_weighted_sum, fusion_damped_hybrid, map_score_to_category, get_recommended_action
- xgboost_regressor: XGBoostSeverityEngine
"""

from src.models.severity.fusion_rules import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
    get_recommended_action,
)
from src.models.severity.xgboost_regressor import XGBoostSeverityEngine

__all__ = [
    "fusion_multiplicative",
    "fusion_weighted_sum",
    "fusion_damped_hybrid",
    "map_score_to_category",
    "get_recommended_action",
    "XGBoostSeverityEngine",
]
