"""
Multimodal Severity Fusion Strategies & Categorization (FR6, FR7).

(Compatibility Facade - logic moved to src.models.severity.fusion_rules)
"""

from src.models.severity.fusion_rules import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
    get_recommended_action,
)

__all__ = [
    "fusion_multiplicative",
    "fusion_weighted_sum",
    "fusion_damped_hybrid",
    "map_score_to_category",
    "get_recommended_action",
]
