"""
Learned XGBoost Severity Fusion Engine (FR6, FR8).

(Compatibility Facade - logic modularized into:
 - src.models.severity.xgboost_regressor
 - src.features.vector_builder
 - src.training.xgb_trainer)
"""

from src.features.vector_builder import (
    FEATURE_NAMES,
    DISEASE_TO_ID,
    assemble_feature_vector,
    extract_features_from_dataframe,
)
from src.models.severity.xgboost_regressor import XGBoostSeverityEngine
from src.training.xgb_trainer import train_xgboost

__all__ = [
    "FEATURE_NAMES",
    "DISEASE_TO_ID",
    "assemble_feature_vector",
    "extract_features_from_dataframe",
    "XGBoostSeverityEngine",
    "train_xgboost",
]
