"""
Verification Gate 5: Temporal Persistence, Progression Trends & GRU Forecaster Suite.

Validates:
1. SQLite database store (FR9): record insertion, chronological ordering, filtering.
2. OLS Progression Trend Classifier (FR9):
   - Validates specification Example 1: [25, 38, 52] -> Worsening.
   - Validates specification Example 2: [48, 49, 48] -> Stable.
   - Validates improving trajectory: [60, 45, 30] -> Improving.
3. Cold-Start Reliability Protocol (Section 6.3):
   - n = 1: INITIAL_OBSERVATION, LOW_COLD_START.
   - n = 2: Two-point linear extrapolation, MEDIUM confidence.
   - n >= 3: Full OLS + GRU, HIGH confidence.
4. PyTorch GRU 7-Day Progression Forecaster (FR10):
   - Sliding sequence generation from longitudinal records.
   - Model training, evaluation, and serialization to results/gru_forecaster.pt.
   - Inference engine predicting bounded [0.0, 100.0] future severity.
"""

import os
import pytest
import numpy as np

from src.schemas.contracts import ProgressionTrend, SeverityCategory, DiseaseType
from src.temporal.database import SeverityDatabase
from src.temporal.trend_classifier import calculate_ols_trend, parse_date
from src.temporal.gru_forecaster import (
    SeverityGRU,
    GRUProgressionForecaster,
    train_gru,
    build_sequences_from_df,
)


def test_sqlite_database(tmp_path):
    """Verifies SQLite storage, indexing, and chronological query retrieval."""
    db_file = str(tmp_path / "test_severity.db")
    db = SeverityDatabase(db_path=db_file)

    # Insert sequential observations
    db.insert_record(
        plant_id="PLANT_001",
        zone_id="ZONE_A",
        leaf_id="LEAF_01",
        observation_date="2026-03-01",
        disease_label="black_sigatoka",
        scope_pct=10.5,
        impact_score=28.0,
        urgency_score=65.0,
        severity_score=32.4,
        severity_category="Moderate",
        fusion_method="F_hybrid",
        stover_grade=2
    )

    db.insert_record(
        plant_id="PLANT_001",
        zone_id="ZONE_A",
        leaf_id="LEAF_01",
        observation_date="2026-03-08",
        disease_label="black_sigatoka",
        scope_pct=18.2,
        impact_score=42.0,
        urgency_score=70.0,
        severity_score=45.1,
        severity_category="Moderate",
        fusion_method="F_hybrid",
        stover_grade=3
    )

    # Insert another plant
    db.insert_record(
        plant_id="PLANT_002",
        zone_id="ZONE_B",
        leaf_id="LEAF_01",
        observation_date="2026-03-01",
        disease_label="healthy",
        scope_pct=0.0,
        impact_score=0.0,
        urgency_score=50.0,
        severity_score=0.0,
        severity_category="Low",
        fusion_method="F_hybrid",
        stover_grade=0
    )

    assert db.count_plant_observations("PLANT_001") == 2
    assert db.count_plant_observations("PLANT_002") == 1

    history = db.get_plant_history("PLANT_001")
    assert len(history) == 2
    assert history[0]["observation_date"] == "2026-03-01"
    assert history[1]["observation_date"] == "2026-03-08"
    assert history[0]["severity_score"] == 32.4
    assert history[1]["severity_score"] == 45.1


def test_ols_trend_classifier_specification_examples():
    """
    CRITICAL SPECIFICATION TEST:
    Verifies that the OLS trend classifier precisely matches the specification examples:
    - Example 1: [25, 38, 52] -> Worsening
    - Example 2: [48, 49, 48] -> Stable
    - Improving: [60, 45, 30] -> Improving
    """
    dates_3w = ["2026-03-01", "2026-03-08", "2026-03-15"]

    # Specification Example 1: [25, 38, 52]
    res1 = calculate_ols_trend(dates_3w, [25.0, 38.0, 52.0], tolerance_band_7d=3.0)
    assert res1["progression_trend"] == ProgressionTrend.WORSENING, f"Expected Worsening, got {res1['progression_trend']}"
    assert res1["weekly_rate_of_change"] > 3.0
    assert abs(res1["weekly_rate_of_change"] - 13.5) < 0.2
    assert res1["confidence_flag"] == "HIGH"

    # Specification Example 2: [48, 49, 48]
    res2 = calculate_ols_trend(dates_3w, [48.0, 49.0, 48.0], tolerance_band_7d=3.0)
    assert res2["progression_trend"] == ProgressionTrend.STABLE, f"Expected Stable, got {res2['progression_trend']}"
    assert abs(res2["weekly_rate_of_change"]) <= 3.0
    assert abs(res2["weekly_rate_of_change"] - 0.0) < 0.2
    assert res2["confidence_flag"] == "HIGH"

    # Improving trajectory: [60, 45, 30]
    res3 = calculate_ols_trend(dates_3w, [60.0, 45.0, 30.0], tolerance_band_7d=3.0)
    assert res3["progression_trend"] == ProgressionTrend.IMPROVING
    assert res3["weekly_rate_of_change"] < -3.0
    assert abs(res3["weekly_rate_of_change"] - (-15.0)) < 0.2
    assert res3["confidence_flag"] == "HIGH"


def test_cold_start_protocol():
    """Verifies n=1, n=2 cold-start reliability behavior."""
    # n = 1: Single observation
    res_n1 = calculate_ols_trend(["2026-03-01"], [35.0])
    assert res_n1["progression_trend"] == ProgressionTrend.INITIAL_OBSERVATION
    assert res_n1["weekly_rate_of_change"] is None
    assert res_n1["confidence_flag"] == "LOW_COLD_START"
    assert res_n1["forecast_7d_score"] is None

    # n = 2: Two observations -> Medium confidence linear extrapolation
    res_n2 = calculate_ols_trend(["2026-03-01", "2026-03-08"], [30.0, 40.0])
    assert res_n2["progression_trend"] == ProgressionTrend.WORSENING
    assert res_n2["confidence_flag"] == "MEDIUM"
    assert res_n2["weekly_rate_of_change"] == 10.0
    # Extrapolated score: 40 + 10 = 50.0
    assert res_n2["forecast_7d_score"] == 50.0
    assert res_n2["forecast_7d_category"] == SeverityCategory.HIGH


def test_gru_forecaster_training_and_inference(tmp_path):
    """Verifies GRU temporal forecaster training, checkpoint saving, and 7-day inference."""
    weights_path = str(tmp_path / "test_gru.pt")

    # 1. Train GRU forecaster on synthetic dataset
    results = train_gru(
        metadata_path="data/synthetic/metadata.csv",
        output_weights_path=weights_path,
        epochs=25,
        batch_size=16,
        lr=0.003
    )

    assert os.path.exists(weights_path), "GRU weights file was not created"
    assert "gru_test_rmse" in results
    assert results["gru_test_rmse"] < 15.0, f"GRU test RMSE is too high: {results['gru_test_rmse']}"

    # 2. Operational inference test
    forecaster = GRUProgressionForecaster(model_path=weights_path)

    sample_history = [
        {"severity_score": 25.0, "urgency_score": 60.0, "scope_pct": 18.0, "disease_label": "black_sigatoka"},
        {"severity_score": 34.0, "urgency_score": 65.0, "scope_pct": 25.0, "disease_label": "black_sigatoka"},
        {"severity_score": 45.0, "urgency_score": 70.0, "scope_pct": 35.0, "disease_label": "black_sigatoka"},
    ]

    pred_score, pred_cat = forecaster.predict_7d(sample_history)
    assert 0.0 <= pred_score <= 100.0
    assert isinstance(pred_cat, SeverityCategory)
    assert pred_score > 30.0
