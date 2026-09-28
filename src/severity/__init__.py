"""
Severity Assessment & Multimodal Fusion modules for Component 3 (FR4, FR6, FR7, FR8).
"""

from src.severity.impact_calculators import (
    calculate_sigatoka_impact,
    calculate_stover_gauhl_dsi,
    calculate_cordana_impact,
    calculate_bbtv_impact,
    calculate_disease_impact,
)
from src.severity.fusion_strategies import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
    get_recommended_action,
)
from src.severity.xgboost_engine import (
    assemble_feature_vector,
    extract_features_from_dataframe,
    XGBoostSeverityEngine,
    train_xgboost,
    FEATURE_NAMES,
)

__all__ = [
    "calculate_sigatoka_impact",
    "calculate_stover_gauhl_dsi",
    "calculate_cordana_impact",
    "calculate_bbtv_impact",
    "calculate_disease_impact",
    "fusion_multiplicative",
    "fusion_weighted_sum",
    "fusion_damped_hybrid",
    "map_score_to_category",
    "get_recommended_action",
    "assemble_feature_vector",
    "extract_features_from_dataframe",
    "XGBoostSeverityEngine",
    "train_xgboost",
    "FEATURE_NAMES",
]
