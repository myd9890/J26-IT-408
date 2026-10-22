"""
Feature Engineering Layer for Component 3.

Contains:
- morphometrics: MorphometricFeatures & extract_leaf_and_lesion_features
- impact_scales: Stover-Gauhl, Cordana HB, BBTV, and unified calculate_disease_impact
- vector_builder: 8-D XGBoost tabular feature vector & 5-D sliding temporal sequence builders
"""

from src.features.morphometrics import (
    MorphometricFeatures,
    extract_leaf_and_lesion_features,
)
from src.features.impact_scales import (
    calculate_sigatoka_impact,
    calculate_stover_gauhl_dsi,
    calculate_cordana_impact,
    calculate_bbtv_impact,
    calculate_disease_impact,
)
from src.features.vector_builder import (
    DISEASE_TO_ID,
    FEATURE_NAMES,
    assemble_feature_vector,
    extract_features_from_dataframe,
    build_sequences_from_df,
)

__all__ = [
    "MorphometricFeatures",
    "extract_leaf_and_lesion_features",
    "calculate_sigatoka_impact",
    "calculate_stover_gauhl_dsi",
    "calculate_cordana_impact",
    "calculate_bbtv_impact",
    "calculate_disease_impact",
    "DISEASE_TO_ID",
    "FEATURE_NAMES",
    "assemble_feature_vector",
    "extract_features_from_dataframe",
    "build_sequences_from_df",
]
