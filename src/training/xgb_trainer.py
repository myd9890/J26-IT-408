"""
XGBoost Severity Fusion Training Routine (FR6, FR8).

Executes:
1. Plant-level partitioned training on longitudinal dataset.
2. Fits XGBRegressor on 8-D multimodal feature vectors.
3. Model evaluation against ground truth expert ratings (calculating RMSE, MAE, R^2).
4. Saves serialized model to JSON format.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.data.dataset import plant_level_split
from src.features.vector_builder import extract_features_from_dataframe
from src.models.severity.fusion_rules import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
)
from src.utils.config import get_config


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
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree: Hyperparameters.
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

    print(f"[XGBoost Train] Split sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

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
