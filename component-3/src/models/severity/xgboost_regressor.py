"""
XGBoost Severity Regressor Model Wrapper (FR6).

Loads serialized model and performs real-time severity scoring for 8-D multimodal feature vectors.
"""

import os
from typing import Optional
import numpy as np
import xgboost as xgb


class XGBoostSeverityEngine:
    """
    Trained XGBoost regressor for Multimodal Foliar Disease Severity Assessment.
    """

    def __init__(self, model_path: Optional[str] = "results/xgboost_model.json"):
        self.model_path = model_path
        self.model: Optional[xgb.XGBRegressor] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str):
        """Loads serialized XGBoost model from JSON format."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.model_path = model_path

    def save(self, model_path: str):
        """Saves model to JSON file format."""
        if self.model is None:
            raise ValueError("No model to save. Train before saving.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        self.model_path = model_path

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Predicts fused severity score for a single 8-D feature vector.

        Args:
            feature_vector: numpy array of shape (8,) or (1, 8).

        Returns:
            Scalar severity score in [0.0, 100.0].
        """
        if self.model is None:
            raise RuntimeError("XGBoost model is not loaded. Train or load first.")

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        raw_pred = float(self.model.predict(feature_vector)[0])
        return round(float(np.clip(raw_pred, 0.0, 100.0)), 2)

    def predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Predicts severity scores for batch of features of shape (N, 8).
        """
        if self.model is None:
            raise RuntimeError("XGBoost model is not loaded.")
        raw_preds = self.model.predict(feature_matrix)
        return np.clip(raw_preds, 0.0, 100.0)
