"""
Verification Gate 4: Disease-Specific Impact Scales, Multimodal Fusion & XGBoost Suite.

Validates:
1. Sigatoka 7-band Stover-Gauhl piecewise-linear continuous impact & discrete grades (FR4).
2. Stover-Gauhl Plant Disease Severity Index (DSI) multi-leaf aggregation.
3. Cordana morphometric coalescence proxy impact scoring (FR4).
4. BBTV 5-stage symptom proxy and lamina distortion impact scoring (FR4).
5. All four multimodal fusion formulations (F_mult, F_WS, F_hybrid, F_XGB) (FR6).
6. Multiplicative zero-collapse demonstration vs Damped Hybrid robustness.
7. Agronomic severity categorization thresholds:
   - Validates that score 72.0 strictly maps to SeverityCategory.HIGH.
8. XGBoost 8-D feature vector extraction, training, evaluation, and JSON persistence (FR8).
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.schemas.contracts import DiseaseType, SeverityCategory
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


def test_sigatoka_impact_scales():
    """Verifies Pinzon-Nunez SADs continuous impact and Stover-Gauhl ordinal grades."""
    # Grade 0: S = 0%
    grade0, imp0 = calculate_sigatoka_impact(0.0)
    assert grade0 == 0
    assert imp0 == 0.0

    # Grade 1: 0 < S <= 5.0%
    grade1, imp1 = calculate_sigatoka_impact(2.5)
    assert grade1 == 1
    assert 0.0 < imp1 < 16.67
    assert round(imp1, 1) == 8.3

    # Grade boundary S = 5.0%
    grade_b1, imp_b1 = calculate_sigatoka_impact(5.0)
    assert grade_b1 == 1
    assert abs(imp_b1 - 16.67) < 0.1

    # Grade 2: 5 < S <= 13.0%
    grade2, imp2 = calculate_sigatoka_impact(13.0)
    assert grade2 == 2
    assert abs(imp2 - 33.34) < 0.1

    # Grade 3: 13 < S <= 23.0%
    grade3, imp3 = calculate_sigatoka_impact(23.0)
    assert grade3 == 3
    assert abs(imp3 - 50.0) < 0.1

    # Grade 4: 23 < S <= 40.0%
    grade4, imp4 = calculate_sigatoka_impact(40.0)
    assert grade4 == 4
    assert abs(imp4 - 66.67) < 0.1

    # Grade 5: 40 < S <= 65.0%
    grade5, imp5 = calculate_sigatoka_impact(65.0)
    assert grade5 == 5
    assert abs(imp5 - 83.34) < 0.1

    # Grade 6: 65 < S <= 100.0%
    grade6, imp6 = calculate_sigatoka_impact(100.0)
    assert grade6 == 6
    assert abs(imp6 - 100.0) < 0.1

    # Monotonicity test across range [0, 100]
    scopes = np.linspace(0.0, 100.0, 101)
    impacts = [calculate_sigatoka_impact(s)[1] for s in scopes]
    assert all(impacts[i] <= impacts[i+1] for i in range(len(impacts)-1)), "Impact must be monotonically non-decreasing"


def test_stover_gauhl_dsi():
    """Verifies whole-plant multi-leaf DSI aggregation formula."""
    # Single leaf tests
    assert calculate_stover_gauhl_dsi([0]) == 0.0
    assert calculate_stover_gauhl_dsi([6]) == 100.0
    assert calculate_stover_gauhl_dsi([3]) == 50.0

    # Multi-leaf: 6 leaves with grades [1, 2, 3, 2, 1, 0] -> sum = 9
    # DSI = (9 / (6 * 6)) * 100 = (9 / 36) * 100 = 25.0%
    assert calculate_stover_gauhl_dsi([1, 2, 3, 2, 1, 0]) == 25.0

    # Empty leaves handling
    assert calculate_stover_gauhl_dsi([]) == 0.0


def test_cordana_impact_proxy():
    """Verifies Cordana Leaf Spot morphometric coalescence proxy equation."""
    # Healthy zero scope
    assert calculate_cordana_impact(0.0, 0.0, 0) == 0.0

    # Low scope, low coalescence
    low_imp = calculate_cordana_impact(scope_pct=4.0, coalescence_ratio_pct=2.0, num_lesions=3)
    assert 0.0 < low_imp < 30.0

    # High coalescence and lesions increases score significantly
    high_coal_imp = calculate_cordana_impact(scope_pct=25.0, coalescence_ratio_pct=25.0, num_lesions=14)
    assert high_coal_imp > 60.0

    # Bounded in [0.0, 100.0]
    max_imp = calculate_cordana_impact(scope_pct=90.0, coalescence_ratio_pct=80.0, num_lesions=50)
    assert max_imp <= 100.0


def test_bbtv_symptom_proxy():
    """Verifies BBTV 5-stage symptom proxy and lamina stunting formula."""
    # Stage 0: Healthy
    stage0, imp0 = calculate_bbtv_impact(scope_pct=0.5, morph_distortion_idx=5.0)
    assert stage0 == 0
    assert imp0 == 0.0

    # Stage 1: Initial vein streaking (S=4.0%, D_morph=10.0)
    stage1, imp1 = calculate_bbtv_impact(scope_pct=4.0, morph_distortion_idx=10.0)
    assert stage1 == 1
    assert 25.0 <= imp1 <= 40.0

    # Stage 4: Severe bunching & stunting (D_morph >= 50.0 or S >= 45.0%)
    stage4, imp4 = calculate_bbtv_impact(scope_pct=50.0, morph_distortion_idx=60.0)
    assert stage4 == 4
    assert imp4 == 100.0


def test_multimodal_fusion_formulas():
    """
    Verifies mathematical behavior of F_mult, F_WS, and F_hybrid.
    Specifically validates non-zero collapse behavior.
    """
    impact = 80.0
    scope = 50.0

    # Case A: Urgency = 0.0 (Zero transmission weather)
    # F_mult collapses to 0.0 despite 80% impact and 50% necrosis
    f_mult_zero = fusion_multiplicative(impact_score=impact, urgency_score=0.0, scope_pct=scope)
    assert f_mult_zero == 0.0, "F_mult must demonstrate zero-collapse limitation"

    # F_WS retains 53.5: 0.45 * 80 + 0.35 * 50 = 36 + 17.5 = 53.5
    f_ws_zero = fusion_weighted_sum(impact_score=impact, scope_pct=scope, urgency_score=0.0)
    assert f_ws_zero == 53.5

    # F_hybrid preserves biological baseline with 0.70 floor
    # bio = 0.55 * 80 + 0.45 * 50 = 44 + 22.5 = 66.5
    # env_amp = 0.70 + 0.30 * 0.0 = 0.70
    # F_hybrid = 66.5 * 0.70 = 46.55
    f_hyb_zero = fusion_damped_hybrid(impact_score=impact, scope_pct=scope, urgency_score=0.0)
    assert abs(f_hyb_zero - 46.55) < 0.1

    # Case B: Urgency = 100.0 (Extreme risk weather)
    # env_amp = 0.70 + 0.30 = 1.00 -> F_hybrid = 66.50
    f_hyb_max = fusion_damped_hybrid(impact_score=impact, scope_pct=scope, urgency_score=100.0)
    assert abs(f_hyb_max - 66.50) < 0.1


def test_severity_categorization_thresholds():
    """
    CRITICAL SPECIFICATION TEST:
    Verifies severity categories and specifically confirms that score 72.0 maps to High.
    """
    assert map_score_to_category(0.0) == SeverityCategory.LOW
    assert map_score_to_category(19.9) == SeverityCategory.LOW
    assert map_score_to_category(20.0) == SeverityCategory.MODERATE
    assert map_score_to_category(49.9) == SeverityCategory.MODERATE
    assert map_score_to_category(50.0) == SeverityCategory.HIGH

    # MANDATORY PROPOSAL TEST
    assert map_score_to_category(72.0) == SeverityCategory.HIGH, "Score 72.0 MUST map to SeverityCategory.High!"

    assert map_score_to_category(74.9) == SeverityCategory.HIGH
    assert map_score_to_category(75.0) == SeverityCategory.CRITICAL
    assert map_score_to_category(100.0) == SeverityCategory.CRITICAL


def test_precision_action_prescriptions():
    """Verifies tailored agronomic advice generation."""
    act_low = get_recommended_action(SeverityCategory.LOW, DiseaseType.BLACK_SIGATOKA)
    assert "Routine scheduled scouting" in act_low

    act_mod = get_recommended_action(SeverityCategory.MODERATE, DiseaseType.BLACK_SIGATOKA)
    assert "Targeted surgical de-leafing" in act_mod

    act_high = get_recommended_action(SeverityCategory.HIGH, DiseaseType.BLACK_SIGATOKA)
    assert "Priority systemic fungicide spray" in act_high

    act_crit_bbtv = get_recommended_action(SeverityCategory.CRITICAL, DiseaseType.BBTV)
    assert "whole-mat eradication" in act_crit_bbtv


def test_xgboost_engine_integration(tmp_path):
    """Verifies feature extraction, model training, evaluation, and serialization."""
    model_path = str(tmp_path / "test_xgb.json")

    # 1. Feature assembly
    vec = assemble_feature_vector(
        impact_score=60.0,
        scope_pct=25.0,
        urgency_score=70.0,
        disease=DiseaseType.BLACK_SIGATOKA,
        num_lesions=14,
        coalescence_ratio_pct=8.5,
        morph_distortion_idx=0.0,
        leaf_aspect_ratio=3.4
    )
    assert vec.shape == (8,)
    assert vec.dtype == np.float32
    assert len(FEATURE_NAMES) == 8

    # 2. Train XGBoost model
    results = train_xgboost(
        metadata_path="data/synthetic/metadata.csv",
        output_model_path=model_path,
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1
    )

    assert os.path.exists(model_path), "XGBoost model file was not written"
    assert "xgboost_test_rmse" in results
    assert results["xgboost_test_r2"] > 0.80, f"XGBoost R^2 is too low: {results['xgboost_test_r2']}"

    # 3. Reload engine and test prediction
    engine = XGBoostSeverityEngine(model_path=model_path)
    pred_score = engine.predict(vec)
    assert 0.0 <= pred_score <= 100.0
    cat = map_score_to_category(pred_score)
    assert isinstance(cat, SeverityCategory)
