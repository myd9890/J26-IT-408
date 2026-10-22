"""
Publication Learning Curve Generator (O2).

Plots and renders high-resolution (300 DPI) learning curves:
1. results/figures/fig6_unet_learning_curves.png: U-Net Loss and Dice convergence
2. results/figures/fig7_gru_learning_curves.png: GRU Training and Validation MSE
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = "results/figures"


def plot_unet_learning_curves(
    history_path: str = "results/unet_real_history.json",
    output_fig_path: str = "results/figures/fig6_unet_learning_curves.png"
):
    """Generates dual-panel convergence curves for U-Net semantic segmentation."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if not os.path.exists(history_path):
        history_path = "results/unet_training_history.json"
    if not os.path.exists(history_path):
        print(f"[Plot Curves] Warning: No U-Net history JSON found at {history_path}")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = data["epochs"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    val_lesion_dice = data.get("val_lesion_dice", [])
    val_macro_dice = data.get("val_macro_dice", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Panel 1: Loss curves
    ax1.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2.0, markersize=5, label='Train Loss (Dice + CE)')
    ax1.plot(epochs, val_loss, 's--', color='#d62728', linewidth=2.0, markersize=5, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Combined Loss', fontsize=11, fontweight='bold')
    ax1.set_title('U-Net Loss Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True)

    # Panel 2: Metric curves
    if val_lesion_dice and val_macro_dice:
        ax2.plot(epochs, val_lesion_dice, '^-', color='#2ca02c', linewidth=2.0, markersize=5, label='Lesion Dice')
        ax2.plot(epochs, val_macro_dice, 'd-', color='#9467bd', linewidth=2.0, markersize=5, label='Macro Dice (3-Class)')
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Dice Similarity Coefficient', fontsize=11, fontweight='bold')
        ax2.set_title('U-Net Validation Dice Trajectory', fontsize=12, fontweight='bold')
        ax2.set_ylim(0.5, 1.0)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot Curves] Saved U-Net learning curves to {output_fig_path}")


def plot_gru_learning_curves(
    history_path: str = "results/gru_training_history.json",
    output_fig_path: str = "results/figures/fig7_gru_learning_curves.png"
):
    """Generates convergence curve for PyTorch GRU progression forecaster."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if not os.path.exists(history_path):
        print(f"[Plot Curves] Warning: No GRU history JSON found at {history_path}")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = data["epochs"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    best_epoch = data.get("best_epoch", epochs[-1])

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(epochs, train_loss, '-', color='#1f77b4', linewidth=1.8, label='Training MSE')
    ax.plot(epochs, val_loss, '--', color='#ff7f0e', linewidth=1.8, label='Validation MSE')
    ax.axvline(best_epoch, color='#2ca02c', linestyle=':', linewidth=1.5, label=f'Best Checkpoint (Epoch {best_epoch})')

    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Mean Squared Error', fontsize=11, fontweight='bold')
    ax.set_title('GRU Progression Forecaster Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot Curves] Saved GRU learning curves to {output_fig_path}")


if __name__ == "__main__":
    plot_unet_learning_curves()
    plot_gru_learning_curves()
