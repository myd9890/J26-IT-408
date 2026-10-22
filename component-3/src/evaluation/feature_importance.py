"""
XGBoost Feature Importance & Attribution Analysis (O3).

Quantifies feature attribution across Gain, Weight (Frequency), and Cover
to interpret the multimodal disease severity decision surface.
Generates:
1. results/tables/table_feature_importance.csv
2. results/figures/fig5_xgboost_feature_importance.png (300 DPI)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb

from src.severity.xgboost_engine import FEATURE_NAMES

TABLES_DIR = "results/tables"
FIGURES_DIR = "results/figures"


def generate_feature_importance_analysis(
    model_path: str = "results/xgboost_model.json"
) -> pd.DataFrame:
    """
    Extracts feature importance metrics from trained XGBoost model and renders publication figure.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")

    booster = xgb.Booster()
    booster.load_model(model_path)

    # Feature score mappings from Booster
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")

    feature_labels = {
        "f0": ("Biological Impact (I)", "impact_score"),
        "f1": ("Foliar Necrosis Scope % (S)", "scope_pct"),
        "f2": ("Environmental Urgency (U)", "urgency_score"),
        "f3": ("Pathogen Class ID", "disease_id"),
        "f4": ("Lesion Count", "num_lesions"),
        "f5": ("Lesion Coalescence Ratio %", "coalescence_ratio_pct"),
        "f6": ("Morphological Distortion Index", "morph_distortion_idx"),
        "f7": ("Leaf Aspect Ratio (L/W)", "leaf_aspect_ratio"),
    }

    records = []
    for i in range(8):
        f_key = f"f{i}"
        label, col_name = feature_labels.get(f_key, (FEATURE_NAMES[i], FEATURE_NAMES[i]))
        g = float(gain_scores.get(f_key, 0.0))
        w = float(weight_scores.get(f_key, 0.0))
        c = float(cover_scores.get(f_key, 0.0))

        records.append({
            "Feature_Code": f_key,
            "Feature_Name": col_name,
            "Agronomic_Description": label,
            "Gain_Importance": round(g, 4),
            "Split_Weight": int(w),
            "Sample_Cover": round(c, 2)
        })

    df = pd.DataFrame(records)
    # Normalize gain to percentages
    total_gain = df["Gain_Importance"].sum()
    df["Relative_Gain_Pct"] = round((df["Gain_Importance"] / max(1e-6, total_gain)) * 100.0, 2)
    df = df.sort_values(by="Gain_Importance", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(TABLES_DIR, "table_feature_importance.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Feature Importance] Saved importance table to {csv_path}")

    # Generate Publication Figure at 300 DPI
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Invert order for horizontal bar chart (highest on top)
    plot_df = df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    # Color gradient from dark green to gold/amber
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(plot_df)))

    bars = ax.barh(y_pos, plot_df["Relative_Gain_Pct"], color=colors, edgecolor="black", linewidth=0.8, height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Agronomic_Description"], fontsize=10, fontweight="medium")
    ax.set_xlabel("Relative Gain Importance (%)", fontsize=11, fontweight="bold")
    ax.set_title("XGBoost Multimodal Feature Importance (Relative Gain)", fontsize=13, fontweight="bold", pad=12)

    # Value labels on bars
    for bar in bars:
        w_val = bar.get_width()
        ax.text(
            w_val + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w_val:.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold"
        )

    ax.set_xlim(0, max(plot_df["Relative_Gain_Pct"]) * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "fig5_xgboost_feature_importance.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Feature Importance] Saved publication figure to {fig_path}")

    return df


if __name__ == "__main__":
    generate_feature_importance_analysis()
