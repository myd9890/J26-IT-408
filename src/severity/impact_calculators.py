"""
Disease-Specific Impact Scoring & Proxy Scales (FR4).

Implements:
1. Sigatoka (Black & Yellow):
   - 7-band Stover-Gauhl discrete grade b in {0..6}
   - Pinzon-Nunez Standard Area Diagram (SADs) continuous piecewise-linear impact score I_Sigatoka in [0, 100]
   - Multi-leaf Plant Disease Severity Index (DSI) for T leaves:
     DSI = (sum(n_b * b) / (6 * T)) * 100.0
2. Cordana Leaf Spot:
   - Horsfall-Barratt / EPPO base foliar score I_HB(S)
   - Lesion coalescence ratio R_coal = (max_lesion_area / leaf_pixels) * 100.0
   - Normalized lesion count D_count = min(100.0, (num_lesions / 20.0) * 100.0)
   - Composite impact:
     I_Cordana = clip(0.65 * I_HB(S) + 0.20 * min(100.0, 2.5 * R_coal) + 0.15 * D_count, 0.0, 100.0)
3. Banana Bunchy Top Virus (BBTV):
   - 5-stage symptom proxy (Stage 0: Healthy, 1: Vein Streaking, 2: Marginal Chlorosis,
     3: Leaf Shortening, 4: Bunching & Stunt)
   - Lamina shortening / distortion index D_morph
   - Continuous impact:
     I_BBTV = clip(25.0 + 0.45 * min(100.0, 2.0 * S) + 0.30 * D_morph, 0.0, 100.0)
     (0.0 if asymptomatic healthy)
4. Healthy Control:
   - Fixed I_Healthy = 0.0, Stover-Gauhl Grade = 0
"""

from typing import Tuple, List, Optional
import numpy as np

from src.schemas.contracts import DiseaseType


def calculate_sigatoka_impact(scope_pct: float) -> Tuple[int, float]:
    """
    Computes Stover-Gauhl discrete grade and continuous piecewise-linear impact score
    for Black Sigatoka and Yellow Sigatoka across Pinzon-Nunez SADs bands.

    Bands:
      Grade 0: S == 0.0%       -> I = 0.00
      Grade 1: 0.0% < S <= 5.0% -> I = 0.00 + 16.67 * ((S - 0.0) / 5.0)
      Grade 2: 5.0% < S <= 13.0% -> I = 16.67 + 16.67 * ((S - 5.0) / 8.0)
      Grade 3: 13.0% < S <= 23.0% -> I = 33.33 + 16.67 * ((S - 13.0) / 10.0)
      Grade 4: 23.0% < S <= 40.0% -> I = 50.00 + 16.67 * ((S - 23.0) / 17.0)
      Grade 5: 40.0% < S <= 65.0% -> I = 66.67 + 16.67 * ((S - 40.0) / 25.0)
      Grade 6: 65.0% < S <= 100.0% -> I = 83.33 + 16.67 * ((S - 65.0) / 35.0)

    Args:
        scope_pct: Foliar lesion coverage in [0.0, 100.0].

    Returns:
        (stover_grade, impact_score) where grade in [0..6] and impact in [0.0, 100.0].
    """
    s = float(np.clip(scope_pct, 0.0, 100.0))

    if s <= 0.0:
        return 0, 0.0
    elif s <= 5.0:
        impact = 0.0 + 16.67 * ((s - 0.0) / 5.0)
        return 1, round(float(np.clip(impact, 0.0, 100.0)), 2)
    elif s <= 13.0:
        impact = 16.67 + 16.67 * ((s - 5.0) / 8.0)
        return 2, round(float(np.clip(impact, 0.0, 100.0)), 2)
    elif s <= 23.0:
        impact = 33.33 + 16.67 * ((s - 13.0) / 10.0)
        return 3, round(float(np.clip(impact, 0.0, 100.0)), 2)
    elif s <= 40.0:
        impact = 50.00 + 16.67 * ((s - 23.0) / 17.0)
        return 4, round(float(np.clip(impact, 0.0, 100.0)), 2)
    elif s <= 65.0:
        impact = 66.67 + 16.67 * ((s - 40.0) / 25.0)
        return 5, round(float(np.clip(impact, 0.0, 100.0)), 2)
    else:
        impact = 83.33 + 16.67 * ((s - 65.0) / 35.0)
        return 6, round(float(np.clip(impact, 0.0, 100.0)), 2)


def calculate_stover_gauhl_dsi(grades: List[int]) -> float:
    """
    Computes whole-plant Disease Severity Index (DSI) for T leaves:
    DSI = (sum(n_b * b) / (6 * T)) * 100.0

    Args:
        grades: List of Stover-Gauhl grades b in {0..6} for each imaged leaf.

    Returns:
        DSI in [0.0, 100.0].
    """
    if not grades:
        return 0.0

    total_t = len(grades)
    grade_sum = sum(int(np.clip(g, 0, 6)) for g in grades)
    dsi = (grade_sum / (6.0 * total_t)) * 100.0
    return round(float(np.clip(dsi, 0.0, 100.0)), 2)


def calculate_cordana_impact(
    scope_pct: float,
    coalescence_ratio_pct: float = 0.0,
    num_lesions: int = 0
) -> float:
    """
    Computes Cordana Leaf Spot impact score via morphometric coalescence proxy:
    I_Cordana = clip(0.65 * I_HB(S) + 0.20 * min(100.0, 2.5 * R_coal) + 0.15 * D_count, 0.0, 100.0)

    Args:
        scope_pct: Foliar coverage percentage [0.0, 100.0].
        coalescence_ratio_pct: Area of single largest lesion / leaf area * 100.0.
        num_lesions: Count of valid connected lesion components.

    Returns:
        Impact score in [0.0, 100.0].
    """
    s = float(np.clip(scope_pct, 0.0, 100.0))
    r_coal = float(np.clip(coalescence_ratio_pct, 0.0, 100.0))
    n_les = max(0, int(num_lesions))

    # Base Horsfall-Barratt / EPPO foliar score
    if s <= 0.0:
        i_hb = 0.0
    elif s <= 5.0:
        i_hb = 20.0
    elif s <= 15.0:
        i_hb = 40.0
    elif s <= 30.0:
        i_hb = 60.0
    elif s <= 50.0:
        i_hb = 80.0
    else:
        i_hb = 100.0

    # Coalescence contribution: 2.5 * R_coal capped at 100.0
    coal_term = min(100.0, 2.5 * r_coal)

    # Lesion count contribution: (N / 20) * 100 capped at 100.0
    d_count = min(100.0, (n_les / 20.0) * 100.0)

    impact = 0.65 * i_hb + 0.20 * coal_term + 0.15 * d_count
    return round(float(np.clip(impact, 0.0, 100.0)), 2)


def calculate_bbtv_impact(
    scope_pct: float,
    morph_distortion_idx: float = 0.0,
    is_healthy_label: bool = False
) -> Tuple[int, float]:
    """
    Computes BBTV ordinal symptom stage (0-4) and continuous impact score:
      Stage 0 (Healthy): S < 1.0% and D_morph < 10.0 -> Impact = 0.0
      Stage 1 (Vein Streaking): 1.0% <= S < 8.0% and D_morph < 15.0
      Stage 2 (Marginal Chlorosis): 8.0% <= S < 25.0% or 15.0 <= D_morph < 30.0
      Stage 3 (Leaf Shortening): 25.0% <= S < 45.0% or 30.0 <= D_morph < 50.0
      Stage 4 (Bunching & Stunt): S >= 45.0% or D_morph >= 50.0 -> Impact = 100.0

    Continuous formula for Stages 1-3:
      I_BBTV = clip(25.0 + 0.45 * min(100.0, 2.0 * S) + 0.30 * D_morph, 0.0, 100.0)

    Args:
        scope_pct: Foliar chlorotic / vein streak symptom area percentage.
        morph_distortion_idx: Lamina bunching distortion index [0.0, 100.0].
        is_healthy_label: Explicit healthy classification flag.

    Returns:
        (bbtv_stage, impact_score) where stage in [0..4] and impact in [0.0, 100.0].
    """
    s = float(np.clip(scope_pct, 0.0, 100.0))
    d_morph = float(np.clip(morph_distortion_idx, 0.0, 100.0))

    if is_healthy_label or (s < 1.0 and d_morph < 10.0):
        return 0, 0.0

    # Stage 4: Severe bunching or pervasive symptoms
    if s >= 45.0 or d_morph >= 50.0:
        return 4, 100.0

    # Stage 3: Noticeable shortening
    if (25.0 <= s < 45.0) or (30.0 <= d_morph < 50.0):
        stage = 3
    # Stage 2: Marginal chlorosis
    elif (8.0 <= s < 25.0) or (15.0 <= d_morph < 30.0):
        stage = 2
    # Stage 1: Initial vein streaking
    else:
        stage = 1

    impact = 25.0 + 0.45 * min(100.0, 2.0 * s) + 0.30 * d_morph
    return stage, round(float(np.clip(impact, 0.0, 100.0)), 2)


def calculate_disease_impact(
    disease: DiseaseType,
    scope_pct: float,
    num_lesions: int = 0,
    coalescence_ratio_pct: float = 0.0,
    morph_distortion_idx: float = 0.0
) -> Tuple[float, Optional[int]]:
    """
    Unified dispatcher for foliar disease impact calculations across all 5 classes.

    Args:
        disease: Canonical DiseaseType enum.
        scope_pct: Foliar lesion coverage in [0.0, 100.0].
        num_lesions: Count of lesion components.
        coalescence_ratio_pct: Area percentage of single largest lesion.
        morph_distortion_idx: Lamina bunching / distortion index.

    Returns:
        (impact_score, stover_grade) where stover_grade is set for Sigatoka, else None.
    """
    if disease == DiseaseType.HEALTHY:
        return 0.0, 0

    elif disease in (DiseaseType.BLACK_SIGATOKA, DiseaseType.YELLOW_SIGATOKA):
        grade, impact = calculate_sigatoka_impact(scope_pct)
        return impact, grade

    elif disease == DiseaseType.CORDANA:
        impact = calculate_cordana_impact(
            scope_pct=scope_pct,
            coalescence_ratio_pct=coalescence_ratio_pct,
            num_lesions=num_lesions
        )
        return impact, None

    elif disease == DiseaseType.BBTV:
        _, impact = calculate_bbtv_impact(
            scope_pct=scope_pct,
            morph_distortion_idx=morph_distortion_idx,
            is_healthy_label=False
        )
        return impact, None

    else:
        # Fallback to scope_pct if unknown disease
        return round(float(np.clip(scope_pct, 0.0, 100.0)), 2), None
