"""
High-Level Production Predictors & Inference Wrappers (FR6, FR10).

Provides:
- GRUProgressionForecaster: Predicts 7-day longitudinal progression trajectories from sequential history.
- XGBoostSeverityEngine: Re-exported from src.models.severity for unified inference interface.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from src.schemas.contracts import DiseaseType, SeverityCategory
from src.models.temporal.gru_net import SeverityGRU
from src.models.severity.fusion_rules import map_score_to_category, fusion_damped_hybrid
from src.features.vector_builder import DISEASE_TO_ID
from src.features.impact_scales import calculate_disease_impact
from src.models.severity.xgboost_regressor import XGBoostSeverityEngine


class GRUProgressionForecaster:
    """
    Operational inference engine for GRU-based 7-day progression forecasting.
    """

    def __init__(self, model_path: Optional[str] = "results/gru_forecaster.pt"):
        self.model = SeverityGRU(input_size=5, hidden_size=32, num_layers=1)
        self.is_loaded = False
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            h_size = checkpoint.get("hidden_size", 32)
            self.model = SeverityGRU(input_size=5, hidden_size=h_size, num_layers=1)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_loaded = True

    def predict_7d(
        self,
        historical_records: List[Dict[str, Any]]
    ) -> Tuple[float, SeverityCategory]:
        """
        Forecasts severity score in 7 days based on plant history.

        Args:
            historical_records: Chronologically sorted list of dicts with keys:
              severity_score, urgency_score, scope_pct, impact_score, disease_label.

        Returns:
            (forecast_7d_score, forecast_7d_category)
        """
        if not historical_records:
            return 0.0, SeverityCategory.LOW

        if not self.is_loaded:
            # Fallback to last known severity score if weights not yet loaded
            last_score = float(historical_records[-1].get("severity_score", 0.0))
            return round(last_score, 2), map_score_to_category(last_score)

        features = []
        for r in historical_records:
            dis_str = str(r.get("disease_label", "healthy")).lower()
            dis_id = float(DISEASE_TO_ID.get(dis_str, 4)) / 4.0
            urg = float(r.get("urgency_score", 0.0))
            scope = float(r.get("ground_truth_scope_pct", r.get("scope_pct", 0.0)))
            aspect = float(r.get("ground_truth_aspect_ratio", r.get("aspect_ratio", 3.5)))
            num_les = int(r.get("num_lesions", 0))
            coal_ratio = float(r.get("coalescence_ratio_pct", r.get("coalescence_ratio", 0.0)))
            d_morph = float(np.clip((3.5 - aspect) / (3.5 - 1.5) * 100.0, 0.0, 100.0))

            dis_enum = DiseaseType(dis_str) if dis_str in [e.value for e in DiseaseType] else DiseaseType.HEALTHY
            impact, _ = calculate_disease_impact(dis_enum, scope, num_les, coal_ratio, d_morph)

            if "severity_score" in r and not pd.isna(r["severity_score"]) and float(r["severity_score"]) > 0.0:
                sev = float(r["severity_score"])
            else:
                sev = fusion_damped_hybrid(impact, scope, urg)

            features.append([sev / 100.0, urg / 100.0, scope / 100.0, impact / 100.0, dis_id])

        seq_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 5)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(seq_tensor).item()  # [0.0, 1.0]

        forecast_score = round(float(np.clip(pred_norm * 100.0, 0.0, 100.0)), 2)
        forecast_cat = map_score_to_category(forecast_score)
        return forecast_score, forecast_cat
