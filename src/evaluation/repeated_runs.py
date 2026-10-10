"""
Repeated Experimental Runs with Multi-Seed Uncertainty Quantification (H2).

Executes repeated experimental trials across 3 independent random seeds [42, 123, 999]
for both XGBoost and GRU, reporting mean ± standard deviation to establish statistical rigor.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.severity.xgboost_engine import train_xgboost
from src.temporal.gru_forecaster import train_gru

TABLES_DIR = "results/tables"


def run_repeated_experiments(seeds: List[int] = [42, 123, 999]) -> Dict[str, pd.DataFrame]:
    """
    Runs repeated trials across multiple random seeds and exports uncertainty summaries.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    print(f"[Repeated Runs] Starting multi-seed evaluation across seeds: {seeds}")

    # 1. XGBoost Repeated Runs
    xgb_records = []
    for s in seeds:
        print(f"  --> Running XGBoost with seed={s}...")
        res = train_xgboost(
            metadata_path="data/synthetic/metadata.csv",
            output_model_path=f"results/xgboost_model_seed_{s}.json",
            seed=s
        )
        xgb_records.append({
            "Seed": s,
            "Test_RMSE": res["xgboost_test_rmse"],
            "Test_MAE": res["xgboost_test_mae"],
            "Test_R2": res["xgboost_test_r2"],
            "Train_RMSE": res["xgboost_train_rmse"]
        })

    xgb_df = pd.DataFrame(xgb_records)
    # Compute summary row
    mean_row = {
        "Seed": "Mean",
        "Test_RMSE": round(float(xgb_df["Test_RMSE"].mean()), 4),
        "Test_MAE": round(float(xgb_df["Test_MAE"].mean()), 4),
        "Test_R2": round(float(xgb_df["Test_R2"].mean()), 4),
        "Train_RMSE": round(float(xgb_df["Train_RMSE"].mean()), 4)
    }
    std_row = {
        "Seed": "Std",
        "Test_RMSE": round(float(xgb_df["Test_RMSE"].std()), 4),
        "Test_MAE": round(float(xgb_df["Test_MAE"].std()), 4),
        "Test_R2": round(float(xgb_df["Test_R2"].std()), 4),
        "Train_RMSE": round(float(xgb_df["Train_RMSE"].std()), 4)
    }
    xgb_summary_df = pd.concat([xgb_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    xgb_path = os.path.join(TABLES_DIR, "table_repeated_runs_severity.csv")
    xgb_summary_df.to_csv(xgb_path, index=False)
    print(f"  [XGBoost] Multi-seed summary saved to {xgb_path}")

    # 2. GRU Repeated Runs
    gru_records = []
    for s in seeds:
        print(f"  --> Running GRU with seed={s}...")
        res = train_gru(
            metadata_path="data/synthetic/metadata.csv",
            output_weights_path=f"results/gru_forecaster_seed_{s}.pt",
            epochs=35,
            seed=s
        )
        gru_records.append({
            "Seed": s,
            "Test_RMSE": res["gru_test_rmse"],
            "Test_MAE": res["gru_test_mae"],
            "Best_Epoch": res["best_epoch"]
        })

    gru_df = pd.DataFrame(gru_records)
    mean_row_gru = {
        "Seed": "Mean",
        "Test_RMSE": round(float(gru_df["Test_RMSE"].mean()), 4),
        "Test_MAE": round(float(gru_df["Test_MAE"].mean()), 4),
        "Best_Epoch": round(float(gru_df["Best_Epoch"].mean()), 1)
    }
    std_row_gru = {
        "Seed": "Std",
        "Test_RMSE": round(float(gru_df["Test_RMSE"].std()), 4),
        "Test_MAE": round(float(gru_df["Test_MAE"].std()), 4),
        "Best_Epoch": round(float(gru_df["Best_Epoch"].std()), 2)
    }
    gru_summary_df = pd.concat([gru_df, pd.DataFrame([mean_row_gru, std_row_gru])], ignore_index=True)
    gru_path = os.path.join(TABLES_DIR, "table_repeated_runs_gru.csv")
    gru_summary_df.to_csv(gru_path, index=False)
    print(f"  [GRU] Multi-seed summary saved to {gru_path}")

    return {
        "xgboost_repeated": xgb_summary_df,
        "gru_repeated": gru_summary_df
    }


if __name__ == "__main__":
    run_repeated_experiments()
