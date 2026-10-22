"""
Disease-Specific Impact Scoring & Proxy Scales (FR4).

(Compatibility Facade - logic moved to src.features.impact_scales)
"""

from src.features.impact_scales import (
    calculate_sigatoka_impact,
    calculate_stover_gauhl_dsi,
    calculate_cordana_impact,
    calculate_bbtv_impact,
    calculate_disease_impact,
)

__all__ = [
    "calculate_sigatoka_impact",
    "calculate_stover_gauhl_dsi",
    "calculate_cordana_impact",
    "calculate_bbtv_impact",
    "calculate_disease_impact",
]
