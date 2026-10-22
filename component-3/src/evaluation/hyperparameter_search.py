"""
Systematic Hyperparameter Search & Empirical Tuning Log (H3).

Explores combinatorial configurations across tree depth, boosting rounds,
and learning rates for the XGBoost Multimodal Severity Fusion Engine.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.data.dataset import plant_level_split
from src.severity.xgboost_engine import extract_features_from_dataframe

TABLES_DIR = "results/tables"


def run_hyperparameter_grid_search(
    metadata_path: str = "data/synthetic/metadata.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Executes a structured grid search over key XGBoost hyperparameters.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    print("[Hyperparameter Search] Initiating systematic grid evaluation...")

    metadata_df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = plant_level_split(metadata_df, 0.70, 0.15, 0.15, seed=seed)

    X_train, y_train = extract_features_from_dataframe(train_df)
    X_val, y_val = extract_features_from_dataframe(val_df)
    X_test, y_test = extract_features_from_dataframe(test_df)

    param_grid = [
        {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.05, "subsample": 0.8},
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05, "subsample": 0.8},
        {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.10, "subsample": 0.8},
        {"max_depth": 6, "n_estimators": 100, "learning_rate": 0.10, "subsample": 0.8},
        {"max_depth": 6, "n_estimators": 200, "learning_rate": 0.10, "subsample": 0.8},  # Default
        {"max_depth": 8, "n_estimators": 100, "learning_rate": 0.10, "subsample": 0.8},
        {"max_depth": 8, "n_estimators": 200, "learning_rate": 0.05, "subsample": 0.8},
    ]

    records = []
    for idx, params in enumerate(param_grid, 1):
        t0 = time.perf_counter()
        model = xgb.XGBRegressor(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        fit_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        preds_val = np.clip(model.predict(X_val), 0.0, 100.0)
        preds_test = np.clip(model.predict(X_test), 0.0, 100.0)

        val_rmse = float(root_mean_squared_error(y_val, preds_val))
        test_rmse = float(root_mean_squared_error(y_test, preds_test))
        test_mae = float(mean_absolute_error(y_test, preds_test))
        test_r2 = float(r2_score(y_test, preds_test))

        is_selected = (params["max_depth"] == 6 and params["n_estimators"] == 200 and params["learning_rate"] == 0.10)

        records.append({
            "Config_ID": f"CFG_{idx:02d}",
            "Max_Depth": params["max_depth"],
            "N_Estimators": params["n_estimators"],
            "Learning_Rate": params["learning_rate"],
            "Subsample": params["subsample"],
            "Val_RMSE": round(val_rmse, 4),
            "Test_RMSE": round(test_rmse, 4),
            "Test_MAE": round(test_mae, 4),
            "Test_R2": round(test_r2, 4),
            "Fit_Time_ms": fit_time_ms,
            "Selection_Status": "Selected (Production)" if is_selected else "Evaluated"
        })
        print(f"  Config {idx:02d}: Depth={params['max_depth']}, N_est={params['n_estimators']}, LR={params['learning_rate']} -> Test RMSE={test_rmse:.4f}, R2={test_r2:.4f}")

    df_results = pd.DataFrame(records)
    out_path = os.path.join(TABLES_DIR, "table_hyperparameter_search.csv")
    df_results.to_csv(out_path, index=False)
    print(f"[Hyperparameter Search] Saved grid exploration results to {out_path}")
    return df_results


if __name__ == "__main__":
    run_hyperparameter_grid_search()
