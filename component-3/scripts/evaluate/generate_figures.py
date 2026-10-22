"""
Publication Figure Generator for Component 3.

Generates:
1. results/figures/fig1_segmentation_comparison.png: 4-panel visual comparison at 300 DPI
2. results/figures/fig2_severity_confusion_matrix.png: 4x4 Confusion Matrix at 300 DPI
3. results/figures/fig3_fusion_zero_collapse_analysis.png: Sensitivity curves demonstrating zero-collapse
4. results/figures/fig4_temporal_forecast_trajectories.png: Longitudinal progression curves
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.schemas.contracts import DiseaseType
from src.models.segmentation.unet import UNet
from src.models.segmentation.classical import ClassicalSegmenter
from src.features.vector_builder import extract_features_from_dataframe
from src.features.impact_scales import calculate_disease_impact
from src.models.severity.fusion_rules import (
    fusion_multiplicative,
    fusion_weighted_sum,
    fusion_damped_hybrid,
    map_score_to_category,
)
from src.models.severity.xgboost_regressor import XGBoostSeverityEngine
from src.inference.predictors import GRUProgressionForecaster

FIGURES_DIR = "results/figures"


def generate_figure1_segmentation_comparison():
    """Generates Figure 1: 4-panel visual comparison at 300 DPI."""
    print("[Benchmark] Generating Figure 1: Segmentation Comparison...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    meta_df = pd.read_csv("data/synthetic/metadata.csv")
    sample_row = meta_df[meta_df["disease_label"] == "black_sigatoka"].iloc[2]

    img_path = os.path.join("data/synthetic", sample_row["image_path"])
    mask_path = os.path.join("data/synthetic", sample_row["mask_path"])

    with Image.open(img_path) as im:
        rgb = np.array(im.convert("RGB"))
    with Image.open(mask_path) as mk:
        gt_mask = np.array(mk)

    classical = ClassicalSegmenter()
    pred_c = classical.predict(rgb)

    ckpt = torch.load("results/unet_weights.pt", map_location="cpu", weights_only=False)
    unet = UNet(n_channels=3, n_classes=3, base_filters=ckpt.get("base_filters", 16))
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        pred_u = unet.predict_mask(tensor).squeeze(0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    cmap = matplotlib.colors.ListedColormap(['#1a1a1a', '#2ca02c', '#d62728'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    axes[0].imshow(rgb)
    axes[0].set_title("(a) Input Foliar Image (RGB)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(gt_mask, cmap=cmap, norm=norm)
    axes[1].set_title("(b) Ground-Truth Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_c, cmap=cmap, norm=norm)
    axes[2].set_title("(c) Classical ExG+CIELAB Baseline", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(pred_u, cmap=cmap, norm=norm)
    axes[3].set_title("(d) PyTorch U-Net (Ours)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    cbar = fig.colorbar(im1, ax=axes, orientation="horizontal", fraction=0.046, pad=0.1, ticks=[0, 1, 2])
    cbar.ax.set_xticklabels(['Background (0)', 'Healthy Lamina (1)', 'Diseased Lesion (2)'], fontsize=10)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig1_segmentation_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  --> Saved {fig_path}")


def generate_figure2_confusion_matrix(test_df: pd.DataFrame):
    """Generates Figure 2: 4x4 Severity Confusion Matrix at 300 DPI."""
    print("[Benchmark] Generating Figure 2: Confusion Matrix...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X_test, y_test = extract_features_from_dataframe(test_df)
    xgb_engine = XGBoostSeverityEngine()
    preds = xgb_engine.predict_batch(X_test)

    cat_labels = ["Low", "Moderate", "High", "Critical"]
    y_true_cats = [map_score_to_category(y).value for y in y_test]
    y_pred_cats = [map_score_to_category(p).value for p in preds]

    cm = confusion_matrix(y_true_cats, y_pred_cats, labels=cat_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=np.round(cm_norm * 100, 1), display_labels=cat_labels)
    disp.plot(cmap="Blues", ax=ax, values_format=".1f", colorbar=True)
    ax.set_title("Multimodal Severity Classification Matrix (%)\n(XGBoost vs Simulated Agro-Expert Truth)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Severity Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Severity Category", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig2_severity_confusion_matrix.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  --> Saved {fig_path}")


def generate_figure3_zero_collapse():
    """Generates Figure 3: Zero-Collapse Sensitivity Curves at 300 DPI."""
    print("[Benchmark] Generating Figure 3: Zero-Collapse Sensitivity Curve...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    urgencies = np.linspace(0.0, 100.0, 101)
    impact = 75.0
    scope = 40.0

    f_mult_vals = [fusion_multiplicative(impact, u, scope) for u in urgencies]
    f_ws_vals = [fusion_weighted_sum(impact, scope, u) for u in urgencies]
    f_hyb_vals = [fusion_damped_hybrid(impact, scope, u) for u in urgencies]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(urgencies, f_mult_vals, 'r--', linewidth=2.2, label=r"$F_{mult}$ (Multiplicative Baseline — Fatal Collapse at $U \to 0$)")
    ax.plot(urgencies, f_ws_vals, 'g-.', linewidth=2.0, label=r"$F_{WS}$ (Weighted-Sum Linear Late Fusion)")
    ax.plot(urgencies, f_hyb_vals, 'b-', linewidth=2.5, label=r"$F_{hybrid}$ (Damped Multiplicative Hybrid — Ours)")

    ax.axvspan(0, 15, color='red', alpha=0.10, label="Zero/Low Urgency Risk Zone ($U < 15$)")
    ax.axhline(59.25, color='gray', linestyle=':', label=r"Pure Biological Severity Floor ($0.55\cdot I + 0.45\cdot S$)")

    ax.set_title("Multimodal Fusion Stress Test: Impact of Environmental Urgency on Foliar Severity", fontsize=11, fontweight="bold")
    ax.set_xlabel("Component 2 Environmental Transmission Urgency ($U \in [0, 100]$)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Composite Severity Score ($Y \in [0, 100]$)", fontsize=10, fontweight="bold")
    ax.set_ylim(-2, 85)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig3_fusion_zero_collapse_analysis.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  --> Saved {fig_path}")


def generate_figure4_trajectories():
    """Generates Figure 4: Longitudinal Trajectories at 300 DPI."""
    print("[Benchmark] Generating Figure 4: Longitudinal Progression Trajectories...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    meta_df = pd.read_csv("data/synthetic/metadata.csv")
    forecaster = GRUProgressionForecaster()

    plant_w = meta_df[meta_df["trajectory_type"] == "Worsening"]["plant_id"].iloc[0]
    plant_s = meta_df[meta_df["trajectory_type"] == "Stable"]["plant_id"].iloc[0]
    plant_i = meta_df[meta_df["trajectory_type"] == "Improving"]["plant_id"].iloc[0]

    selected = [("Worsening Cohort", plant_w, "#d62728"),
                ("Stable Cohort", plant_s, "#1f77b4"),
                ("Improving Cohort", plant_i, "#2ca02c")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    for idx, (title, pid, col) in enumerate(selected):
        p_df = meta_df[meta_df["plant_id"] == pid].sort_values("day_index").reset_index(drop=True)
        days = p_df["day_index"].values
        severities = []
        for _, r in p_df.iterrows():
            s = float(r["ground_truth_scope_pct"])
            u = float(r["urgency_score"])
            dis = DiseaseType(str(r["disease_label"]).lower())
            imp, _ = calculate_disease_impact(dis, s)
            severities.append(fusion_damped_hybrid(imp, s, u))

        pred_points = [severities[0], severities[1]]
        for k in range(2, len(severities)):
            hist = p_df.iloc[:k].to_dict("records")
            p_score, _ = forecaster.predict_7d(hist)
            pred_points.append(p_score)

        ax = axes[idx]
        ax.plot(days, severities, 'o-', color=col, linewidth=2.2, label="Actual Ground Truth ($Y_t$)")
        ax.plot(days[2:], pred_points[2:], 's--', color="#ff7f0e", linewidth=2.0, label="GRU 7-Day Forecast ($\hat{Y}_{t+7}$)")
        ax.set_title(f"{title} ({pid})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Plantation Inspection Day", fontsize=10)
        ax.set_ylabel("Foliar Severity Score", fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig4_temporal_forecast_trajectories.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  --> Saved {fig_path}")


if __name__ == "__main__":
    generate_figure1_segmentation_comparison()
    meta_df = pd.read_csv("data/synthetic/metadata.csv")
    from src.data.dataset import plant_level_split
    _, _, test_df = plant_level_split(meta_df, 0.70, 0.15, 0.15, seed=42)
    generate_figure2_confusion_matrix(test_df)
    generate_figure3_zero_collapse()
    generate_figure4_trajectories()
