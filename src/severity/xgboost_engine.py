"""
Learned XGBoost Severity Fusion Engine (FR6, FR8).

Implements:
1. 8-dimensional feature vector extraction:
   x = [impact_score, scope_pct, urgency_score, disease_id,
        num_lesions, coalescence_ratio_pct, morph_distortion_idx, leaf_aspect_ratio]
2. Plant-level partitioned training on longitudinal dataset.
3. Model evaluation against ground truth expert ratings (calculating RMSE, MAE, R^2).
4. Model serialization and reloading from results/xgboost_model.json.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.schemas.contracts import DiseaseType
from src.data.dataset import plant_level_split
from src.severity.impact_calculators import calculate_disease_impact
from src.severity.fusion_strategies import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
)
from src.evaluation.expert_simulator import AgroExpertSimulator
from src.utils.config import get_config

FEATURE_NAMES = [
    "impact_score",
    "scope_pct",
    "urgency_score",
    "disease_id",
    "num_lesions",
    "coalescence_ratio_pct",
    "morph_distortion_idx",
    "leaf_aspect_ratio"
]

DISEASE_TO_ID: Dict[str, int] = {
    DiseaseType.BLACK_SIGATOKA.value: 0,
    DiseaseType.YELLOW_SIGATOKA.value: 1,
    DiseaseType.CORDANA.value: 2,
    DiseaseType.BBTV.value: 3,
    DiseaseType.HEALTHY.value: 4,
}


def assemble_feature_vector(
    impact_score: float,
    scope_pct: float,
    urgency_score: float,
    disease: Union[DiseaseType, str],
    num_lesions: int = 0,
    coalescence_ratio_pct: float = 0.0,
    morph_distortion_idx: float = 0.0,
    leaf_aspect_ratio: float = 3.5
) -> np.ndarray:
    """
    Constructs the 8-dimensional feature vector for XGBoost fusion.

    Args:
        impact_score: Biological impact score in [0.0, 100.0].
        scope_pct: Foliar scope coverage in [0.0, 100.0].
        urgency_score: Environmental transmission urgency in [0.0, 100.0].
        disease: DiseaseType enum or string representation.
        num_lesions: Count of lesion components.
        coalescence_ratio_pct: Area percentage of largest single lesion.
        morph_distortion_idx: Lamina bunching distortion index.
        leaf_aspect_ratio: Minimum-area bounding box length-to-width ratio.

    Returns:
        1D numpy array of shape (8,) with float32 values.
    """
    dis_str = disease.value if isinstance(disease, DiseaseType) else str(disease).lower()
    disease_id = float(DISEASE_TO_ID.get(dis_str, 4))

    vec = np.array([
        float(np.clip(impact_score, 0.0, 100.0)),
        float(np.clip(scope_pct, 0.0, 100.0)),
        float(np.clip(urgency_score, 0.0, 100.0)),
        disease_id,
        float(max(0, num_lesions)),
        float(np.clip(coalescence_ratio_pct, 0.0, 100.0)),
        float(np.clip(morph_distortion_idx, 0.0, 100.0)),
        float(max(1.0, leaf_aspect_ratio))
    ], dtype=np.float32)

    return vec


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


def extract_features_from_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 8-D feature matrix X and target severity array y from dataset DataFrame.
    """
    X_rows = []
    y_vals = []

    for row_idx, row in df.iterrows():
        disease_str = str(row["disease_label"]).lower()
        try:
            disease_enum = DiseaseType(disease_str)
        except ValueError:
            disease_enum = DiseaseType.HEALTHY

        scope_pct = float(row.get("ground_truth_scope_pct", row.get("scope_pct", 0.0)))
        urgency_score = float(row.get("urgency_score", 0.0))
        num_lesions = int(row.get("num_lesions", 0))
        coalescence_ratio = float(row.get("coalescence_ratio_pct", row.get("coalescence_ratio", 0.0)))
        aspect_ratio = float(row.get("ground_truth_aspect_ratio", row.get("aspect_ratio", 3.5)))

        # Compute morph_distortion_idx
        d_morph = float(np.clip((3.5 - aspect_ratio) / (3.5 - 1.5) * 100.0, 0.0, 100.0))

        # Biological impact
        impact, _ = calculate_disease_impact(
            disease=disease_enum,
            scope_pct=scope_pct,
            num_lesions=num_lesions,
            coalescence_ratio_pct=coalescence_ratio,
            morph_distortion_idx=d_morph
        )

        vec = assemble_feature_vector(
            impact_score=impact,
            scope_pct=scope_pct,
            urgency_score=urgency_score,
            disease=disease_enum,
            num_lesions=num_lesions,
            coalescence_ratio_pct=coalescence_ratio,
            morph_distortion_idx=d_morph,
            leaf_aspect_ratio=aspect_ratio
        )
        X_rows.append(vec)

        # Target severity: use ground truth if present; otherwise simulate expert inspection rating
        if "severity_score" in row and not pd.isna(row["severity_score"]) and float(row["severity_score"]) > 0.0:
            target_sev = float(row["severity_score"])
        elif "expert_severity" in row and not pd.isna(row["expert_severity"]) and float(row["expert_severity"]) > 0.0:
            target_sev = float(row["expert_severity"])
        else:
            base_sev = fusion_damped_hybrid(impact, scope_pct, urgency_score)
            s_idx = int(row_idx) if isinstance(row_idx, (int, np.integer)) else 42
            sim = AgroExpertSimulator(sigma_noise=4.2, seed=42 + (s_idx % 1000))
            target_sev = sim.simulate_expert_rating(base_sev)
        y_vals.append(target_sev)

    X = np.vstack(X_rows).astype(np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y


def train_xgboost(
    metadata_path: str = "data/synthetic/metadata.csv",
    output_model_path: str = "results/xgboost_model.json",
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    learning_rate: Optional[float] = None,
    subsample: Optional[float] = None,
    colsample_bytree: Optional[float] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains XGBoost on plant-level training partition and evaluates against test split.

    Args:
        metadata_path: Path to dataset CSV.
        output_model_path: Path to save the trained model JSON.
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree: XGBoost hyperparameters.
        seed: Random seed.

    Returns:
        Dictionary of performance metrics (RMSE, MAE, R2) for train and test splits,
        plus comparison against mathematical baselines F_mult, F_WS, F_hybrid.
    """
    n_estimators = n_estimators or int(get_config("xgboost.n_estimators", 200))
    max_depth = max_depth or int(get_config("xgboost.max_depth", 6))
    learning_rate = learning_rate or float(get_config("xgboost.learning_rate", 0.1))
    subsample = subsample or float(get_config("xgboost.subsample", 0.8))
    colsample_bytree = colsample_bytree or float(get_config("xgboost.colsample_bytree", 0.8))

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = plant_level_split(metadata_df, 0.70, 0.15, 0.15, seed=seed)

    X_train, y_train = extract_features_from_dataframe(train_df)
    X_val, y_val = extract_features_from_dataframe(val_df)
    X_test, y_test = extract_features_from_dataframe(test_df)

    # Instantiate XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Save model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save_model(output_model_path)

    # Predictions
    preds_train = np.clip(model.predict(X_train), 0.0, 100.0)
    preds_test = np.clip(model.predict(X_test), 0.0, 100.0)

    # Baseline predictions on test set
    f_mult_test = [fusion_multiplicative(X_test[i, 0], X_test[i, 2], X_test[i, 1]) for i in range(len(X_test))]
    f_ws_test = [fusion_weighted_sum(X_test[i, 0], X_test[i, 1], X_test[i, 2]) for i in range(len(X_test))]
    f_hybrid_test = [fusion_damped_hybrid(X_test[i, 0], X_test[i, 1], X_test[i, 2]) for i in range(len(X_test))]

    results = {
        "xgboost_train_rmse": round(float(root_mean_squared_error(y_train, preds_train)), 4),
        "xgboost_train_mae": round(float(mean_absolute_error(y_train, preds_train)), 4),
        "xgboost_train_r2": round(float(r2_score(y_train, preds_train)), 4),
        "xgboost_test_rmse": round(float(root_mean_squared_error(y_test, preds_test)), 4),
        "xgboost_test_mae": round(float(mean_absolute_error(y_test, preds_test)), 4),
        "xgboost_test_r2": round(float(r2_score(y_test, preds_test)), 4),
        "f_mult_test_rmse": round(float(root_mean_squared_error(y_test, f_mult_test)), 4),
        "f_ws_test_rmse": round(float(root_mean_squared_error(y_test, f_ws_test)), 4),
        "f_hybrid_test_rmse": round(float(root_mean_squared_error(y_test, f_hybrid_test)), 4),
    }

    print("[XGBoost Train] Evaluation Results:", results)
    return results


if __name__ == "__main__":
    train_xgboost()
