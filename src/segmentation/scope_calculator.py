"""
Morphometric Scope Engine (FR3) for Banana Foliar Disease Assessment.

(Compatibility Facade - logic moved to src.features.morphometrics)
"""

from src.features.morphometrics import (
    MorphometricFeatures,
    extract_leaf_and_lesion_features,
)

__all__ = [
    "MorphometricFeatures",
    "extract_leaf_and_lesion_features",
]
