"""
Multimodal Severity Fusion Strategies & Categorization (FR6, FR7).

Implements the four benchmarked fusion formulations:
1. Multiplicative Baseline (F_mult):
   F_mult = (I * U * S) / 10000.0
   (Collapses to 0 when U=0 even if leaf is completely necrotic).
2. Weighted-Sum Late Fusion (F_WS):
   F_WS = 0.45 * I + 0.35 * S + 0.20 * U
   (Linear non-collapsing benchmark).
3. Damped Multiplicative Hybrid (F_hybrid - Recommended Operational Formula):
   F_hybrid = (0.55 * I + 0.45 * S) * (0.70 + 0.30 * (U / 100.0))
   (Preserves biological severity floor while amplifying by microclimate urgency).
4. Learned XGBoost Fusion (F_XGB):
   Interface to gradient boosted regressor trained on 8-D morphometric feature vector.

Also implements:
- map_score_to_category: Maps severity score in [0.0, 100.0] into Low, Moderate, High, Critical.
- get_recommended_action: Maps severity category and disease into actionable precision agronomic advice.
"""

from typing import Dict, Any, Union
import numpy as np

from src.schemas.contracts import DiseaseType, SeverityCategory


def fusion_multiplicative(impact_score: float, urgency_score: float, scope_pct: float) -> float:
    """
    Multiplicative Baseline (F_mult):
    F_mult = (I * U * S) / 10,000.0

    Note: Demonstrates severe underestimation collapse if environmental urgency U -> 0
    even when foliar necrosis is extensive.
    """
    i = float(np.clip(impact_score, 0.0, 100.0))
    u = float(np.clip(urgency_score, 0.0, 100.0))
    s = float(np.clip(scope_pct, 0.0, 100.0))

    score = (i * u * s) / 10000.0
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def fusion_weighted_sum(
    impact_score: float,
    scope_pct: float,
    urgency_score: float,
    w_impact: float = 0.45,
    w_scope: float = 0.35,
    w_urgency: float = 0.20
) -> float:
    """
    Weighted-Sum Late Fusion (F_WS):
    F_WS = w_impact * I + w_scope * S + w_urgency * U

    Weights default to 0.45, 0.35, 0.20 as calibrated in blueprint.
    """
    i = float(np.clip(impact_score, 0.0, 100.0))
    s = float(np.clip(scope_pct, 0.0, 100.0))
    u = float(np.clip(urgency_score, 0.0, 100.0))

    score = w_impact * i + w_scope * s + w_urgency * u
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def fusion_damped_hybrid(
    impact_score: float,
    scope_pct: float,
    urgency_score: float,
    w_impact: float = 0.55,
    w_scope: float = 0.45,
    env_floor: float = 0.70,
    env_scale: float = 0.30
) -> float:
    """
    Damped Multiplicative Hybrid (F_hybrid — Recommended Operational Strategy):
    F_hybrid = (0.55 * I + 0.45 * S) * (0.70 + 0.30 * (U / 100.0))

    Guarantees that biological severity is preserved at >= 70% even in zero-risk
    weather, while high transmission urgency amplifies severity up to 100%.
    """
    i = float(np.clip(impact_score, 0.0, 100.0))
    s = float(np.clip(scope_pct, 0.0, 100.0))
    u = float(np.clip(urgency_score, 0.0, 100.0))

    bio_severity = w_impact * i + w_scope * s
    env_amplifier = env_floor + env_scale * (u / 100.0)

    score = bio_severity * env_amplifier
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def map_score_to_category(score: float) -> SeverityCategory:
    """
    Categorizes severity score into validated agronomic bands (Section 2.2 / Section 5):
      [0.0, 20.0)   -> Low
      [20.0, 50.0)  -> Moderate
      [50.0, 75.0)  -> High      (Validates: 72.0 maps to High)
      [75.0, 100.0] -> Critical

    Args:
        score: Severity score in [0.0, 100.0].

    Returns:
        SeverityCategory enum member.
    """
    clipped = float(np.clip(score, 0.0, 100.0))

    if clipped < 20.0:
        return SeverityCategory.LOW
    elif clipped < 50.0:
        return SeverityCategory.MODERATE
    elif clipped < 75.0:
        return SeverityCategory.HIGH
    else:
        return SeverityCategory.CRITICAL


def get_recommended_action(category: SeverityCategory, disease: DiseaseType) -> str:
    """
    Prescribes targeted precision agronomic interventions based on severity category
    and specific pathogen biology.

    Args:
        category: Classified SeverityCategory.
        disease: Diagnosed DiseaseType.

    Returns:
        Actionable instruction string for Component 4.
    """
    if category == SeverityCategory.LOW:
        return "Routine scheduled scouting; no chemical intervention."

    elif category == SeverityCategory.MODERATE:
        if disease == DiseaseType.BBTV:
            return "Flag mat for quarantine observation; apply vector aphicide (imidacloprid)."
        return "Targeted surgical de-leafing of necrotic tips; preventive bio-fungicide."

    elif category == SeverityCategory.HIGH:
        if disease == DiseaseType.BBTV:
            return "Strict mat quarantine; spray aphicide immediately; prep for containment."
        return "Priority systemic fungicide spray & drainage check."

    else:  # CRITICAL
        if disease == DiseaseType.BBTV:
            return "Immediate quarantine, whole-leaf pruning, and whole-mat eradication to prevent aphid vector transmission."
        return "Immediate quarantine, whole-leaf pruning, or mat eradication."
