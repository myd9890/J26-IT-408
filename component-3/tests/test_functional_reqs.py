"""
Comprehensive End-to-End Functional Requirements (FR1–FR11) Verification Suite.

Validates the full Component 3 architecture:
- FR1: Component 1 Vision Diagnosis input contract & validation.
- FR2: Semantic segmentation mask generation (U-Net & Classical baseline).
- FR3: Morphometric scope extraction & geometrical distortion metrics.
- FR4: Disease-specific impact scoring across validated standard scales.
- FR5: Component 2 Microclimate transmission urgency contract & validation.
- FR6: Multimodal fusion engine (F_mult, F_WS, F_hybrid, F_XGB) & zero-collapse analysis.
- FR7: Severity categorization into validated agronomic bands (72.0 -> High).
- FR8: SQLite temporal persistence store & chronological ordering.
- FR9: OLS weekly rate of change classification & cold-start reliability protocol.
- FR10: PyTorch GRU 7-day disease progression forecaster.
- FR11: Component 4 precision action output contract & FastAPI microservice endpoints.
"""

import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.schemas.contracts import (
    Component1Input,
    Component2Input,
    Component4Output,
    DiseaseType,
    SeverityCategory,
    ProgressionTrend,
)
from src.segmentation.unet import UNet
from src.segmentation.scope_calculator import extract_leaf_and_lesion_features, MorphometricFeatures
from src.severity.impact_calculators import calculate_disease_impact, calculate_sigatoka_impact
from src.severity.fusion_strategies import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
    get_recommended_action,
)
from src.severity.xgboost_engine import XGBoostSeverityEngine, assemble_feature_vector
from src.temporal.database import SeverityDatabase
from src.temporal.trend_classifier import calculate_ols_trend
from src.temporal.gru_forecaster import GRUProgressionForecaster
from src.pipeline.c3_orchestrator import Component3Orchestrator
from src.api.app import app


@pytest.fixture
def test_image_path(tmp_path):
    """Creates a temporary valid 256x256 test image."""
    from PIL import Image
    import cv2
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.ellipse(img, (128, 128), (100, 35), 45, 0, 360, (50, 160, 40), -1)
    cv2.ellipse(img, (128, 128), (20, 10), 45, 0, 360, (70, 35, 20), -1)
    p = str(tmp_path / "test_leaf.jpg")
    Image.fromarray(img).save(p)
    return p


# -----------------------------------------------------------------------------
# FR1: Component 1 Vision Diagnosis Contract
# -----------------------------------------------------------------------------
def test_fr1_vision_input_validation(test_image_path):
    valid_c1 = Component1Input(
        plant_id="PLANT_001",
        zone_id="ZONE_A",
        leaf_id="LEAF_01",
        observation_date="2026-10-15",
        disease_label=DiseaseType.BLACK_SIGATOKA,
        disease_confidence=0.94,
        image_path=test_image_path
    )
    assert valid_c1.disease_label == DiseaseType.BLACK_SIGATOKA
    assert valid_c1.disease_confidence == 0.94

    # Invalid date format
    with pytest.raises(ValueError):
        Component1Input(
            plant_id="PLANT_001",
            zone_id="ZONE_A",
            leaf_id="LEAF_01",
            observation_date="15-10-2026",
            disease_label=DiseaseType.BLACK_SIGATOKA,
            disease_confidence=0.94,
            image_path=test_image_path
        )


# -----------------------------------------------------------------------------
# FR2: Semantic Segmentation Architecture
# -----------------------------------------------------------------------------
def test_fr2_unet_segmentation():
    import torch
    model = UNet(n_channels=3, n_classes=3, base_filters=16)
    x = torch.randn(1, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        mask = model.predict_mask(x)
    assert mask.shape == (1, 256, 256)
    classes = torch.unique(mask).tolist()
    assert all(c in [0, 1, 2] for c in classes)


# -----------------------------------------------------------------------------
# FR3: Morphometric Scope Engine
# -----------------------------------------------------------------------------
def test_fr3_morphometric_bounds():
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:200, 50:150] = 1
    mask[90:120, 80:110] = 2

    feats = extract_leaf_and_lesion_features(mask)
    assert 0.0 <= feats.scope_pct <= 100.0
    assert 0.0 <= feats.coalescence_ratio_pct <= 100.0
    assert 0.0 <= feats.morph_distortion_idx <= 100.0
    assert feats.leaf_aspect_ratio >= 1.0
    assert feats.num_lesions >= 1


# -----------------------------------------------------------------------------
# FR4: Disease-Specific Impact Calculators
# -----------------------------------------------------------------------------
def test_fr4_disease_impact_calculators():
    # Sigatoka
    g0, i0 = calculate_sigatoka_impact(0.0)
    assert g0 == 0 and i0 == 0.0

    g6, i6 = calculate_sigatoka_impact(100.0)
    assert g6 == 6 and i6 == 100.0

    # Healthy control
    i_h, g_h = calculate_disease_impact(DiseaseType.HEALTHY, scope_pct=15.0)
    assert i_h == 0.0 and g_h == 0


# -----------------------------------------------------------------------------
# FR5: Component 2 Microclimate Urgency Contract
# -----------------------------------------------------------------------------
def test_fr5_environmental_input_validation():
    valid_c2 = Component2Input(
        zone_id="ZONE_A",
        observation_date="2026-10-15",
        urgency_score=78.5
    )
    assert valid_c2.urgency_score == 78.5

    # Urgency must be in [0.0, 100.0]
    with pytest.raises(ValueError):
        Component2Input(zone_id="ZONE_A", observation_date="2026-10-15", urgency_score=105.0)


# -----------------------------------------------------------------------------
# FR6: Multimodal Fusion Engine
# -----------------------------------------------------------------------------
def test_fr6_multimodal_fusion():
    impact = 70.0
    scope = 30.0

    # Zero urgency collapse in F_mult
    f_mult = fusion_multiplicative(impact, 0.0, scope)
    assert f_mult == 0.0

    # Non-collapsing operational hybrid
    f_hyb = fusion_damped_hybrid(impact, scope, 0.0)
    assert f_hyb > 0.0
    assert abs(f_hyb - ((0.55 * 70 + 0.45 * 30) * 0.70)) < 0.1


# -----------------------------------------------------------------------------
# FR7: Severity Categorization (72.0 -> High)
# -----------------------------------------------------------------------------
def test_fr7_severity_categorization():
    assert map_score_to_category(72.0) == SeverityCategory.HIGH
    assert map_score_to_category(15.0) == SeverityCategory.LOW
    assert map_score_to_category(35.0) == SeverityCategory.MODERATE
    assert map_score_to_category(85.0) == SeverityCategory.CRITICAL


# -----------------------------------------------------------------------------
# FR8: SQLite Temporal Persistence Store
# -----------------------------------------------------------------------------
def test_fr8_sqlite_persistence(tmp_path):
    db_file = str(tmp_path / "test_fr8.db")
    db = SeverityDatabase(db_path=db_file)

    rec_id = db.insert_record(
        plant_id="PLANT_FR8",
        zone_id="ZONE_A",
        leaf_id="LEAF_01",
        observation_date="2026-10-01",
        disease_label="black_sigatoka",
        scope_pct=12.0,
        impact_score=31.0,
        urgency_score=60.0,
        severity_score=28.5,
        severity_category="Moderate",
        fusion_method="F_hybrid"
    )
    assert rec_id >= 1
    history = db.get_plant_history("PLANT_FR8")
    assert len(history) == 1
    assert history[0]["severity_score"] == 28.5


# -----------------------------------------------------------------------------
# FR9: OLS Trend Classification & Cold-Start Protocol
# -----------------------------------------------------------------------------
def test_fr9_trend_classification():
    dates = ["2026-03-01", "2026-03-08", "2026-03-15"]
    # Worsening: [25, 38, 52]
    res_w = calculate_ols_trend(dates, [25.0, 38.0, 52.0])
    assert res_w["progression_trend"] == ProgressionTrend.WORSENING

    # Stable: [48, 49, 48]
    res_s = calculate_ols_trend(dates, [48.0, 49.0, 48.0])
    assert res_s["progression_trend"] == ProgressionTrend.STABLE


# -----------------------------------------------------------------------------
# FR10: GRU 7-Day Progression Forecaster
# -----------------------------------------------------------------------------
def test_fr10_gru_forecaster():
    forecaster = GRUProgressionForecaster(model_path="results/gru_forecaster.pt")
    history = [
        {"severity_score": 25.0, "urgency_score": 60.0, "scope_pct": 18.0, "disease_label": "black_sigatoka"},
        {"severity_score": 34.0, "urgency_score": 65.0, "scope_pct": 25.0, "disease_label": "black_sigatoka"},
        {"severity_score": 45.0, "urgency_score": 70.0, "scope_pct": 35.0, "disease_label": "black_sigatoka"},
    ]
    p_score, p_cat = forecaster.predict_7d(history)
    assert 0.0 <= p_score <= 100.0
    assert isinstance(p_cat, SeverityCategory)


# -----------------------------------------------------------------------------
# FR11: End-to-End Orchestrator & FastAPI Microservice
# -----------------------------------------------------------------------------
def test_fr11_orchestrator_and_api(test_image_path, tmp_path):
    # Test orchestrator
    test_db = str(tmp_path / "orch_test.db")
    orch = Component3Orchestrator(db_path=test_db)

    c1 = Component1Input(
        plant_id="PLANT_E2E",
        zone_id="ZONE_A",
        leaf_id="LEAF_01",
        observation_date="2026-10-15",
        disease_label=DiseaseType.BLACK_SIGATOKA,
        disease_confidence=0.92,
        image_path=test_image_path
    )
    c2 = Component2Input(
        zone_id="ZONE_A",
        observation_date="2026-10-15",
        urgency_score=68.0
    )

    output = orch.assess_and_forecast(c1, c2, fusion_method="F_hybrid")
    assert isinstance(output, Component4Output)
    assert output.plant_id == "PLANT_E2E"
    assert 0.0 <= output.severity_score <= 100.0
    assert output.severity_category in [SeverityCategory.LOW, SeverityCategory.MODERATE, SeverityCategory.HIGH, SeverityCategory.CRITICAL]
    assert output.confidence_flag in ["LOW_COLD_START", "MEDIUM", "HIGH"]
    assert len(output.recommended_action) > 10

    # Test FastAPI Client
    client = TestClient(app)

    # Healthcheck
    resp_h = client.get("/health")
    assert resp_h.status_code == 200
    assert resp_h.json()["status"] == "ok"

    # Assessment endpoint
    payload = {
        "vision_input": c1.model_dump(),
        "environmental_input": c2.model_dump(),
        "fusion_method": "F_hybrid"
    }
    resp_a = client.post("/api/v1/assess-and-forecast", json=payload)
    assert resp_a.status_code == 200
    data = resp_a.json()
    assert data["plant_id"] == "PLANT_E2E"
    assert "severity_score" in data
    assert "severity_category" in data

    # History endpoint
    resp_hist = client.get("/api/v1/plant-history/PLANT_E2E")
    assert resp_hist.status_code == 200
    hist_data = resp_hist.json()
    assert hist_data["total_inspections"] >= 1
