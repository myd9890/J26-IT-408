"""
U-Net Training and Evaluation Routine for Banana Foliar Disease Segmentation.

Executes:
1. Plant-level partitioned dataset loading (84 train / 18 val / 18 test plants) to ensure zero data leakage.
2. AdamW optimizer with CosineAnnealingLR scheduling.
3. Combined Multi-Class Soft Dice + Cross-Entropy loss.
4. Validation evaluation after each epoch tracking IoU, Dice, Precision, and Recall.
5. Best model checkpointing to results/unet_weights.pt.
6. Exports full training log to results/unet_training_history.json.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import plant_level_split, BananaSegmentationDataset
from src.segmentation.unet import UNet
from src.segmentation.loss_and_metrics import CombinedDiceCrossEntropyLoss, compute_segmentation_metrics
from src.utils.config import get_config


def train_unet(
    metadata_path: str = "data/synthetic/metadata.csv",
    base_dir: str = "data/synthetic",
    output_weights_path: str = "results/unet_weights.pt",
    history_path: str = "results/unet_training_history.json",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    base_filters: Optional[int] = None,
    patience: int = 5,
    device: Optional[str] = None,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains U-Net on the synthetic or real longitudinal cohort using plant-level partitioning.
    Includes early stopping based on validation Macro Dice.

    Args:
        metadata_path: Path to dataset metadata CSV.
        base_dir: Root directory for image/mask references.
        output_weights_path: Path where best model weights will be saved.
        history_path: Path where training history JSON will be saved.
        epochs: Number of training epochs.
        batch_size: DataLoader mini-batch size.
        learning_rate: Initial AdamW learning rate.
        base_filters: Number of initial feature channels in U-Net.
        device: 'cuda', 'cpu', or None (auto-detect).
        seed: Random seed for reproducibility.
        max_train_samples: Optional limit on training samples for rapid testing.
        max_val_samples: Optional limit on validation samples for rapid testing.

    Returns:
        Dictionary containing training history and best validation metrics.
    """
    epochs = epochs or int(get_config("segmentation.unet_epochs", 10))
    batch_size = batch_size or int(get_config("segmentation.unet_batch_size", 8))
    learning_rate = learning_rate or float(get_config("segmentation.unet_learning_rate", 0.001))
    base_filters = base_filters or 32

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    print(f"[UNet Train] Using device: {device_obj}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # 1. Load metadata and split by plant_id
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = plant_level_split(metadata_df, 0.70, 0.15, 0.15, seed=seed)

    if max_train_samples is not None:
        train_df = train_df.iloc[:max_train_samples].copy().reset_index(drop=True)
    if max_val_samples is not None:
        val_df = val_df.iloc[:max_val_samples].copy().reset_index(drop=True)

    print(f"[UNet Train] Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. PyTorch DataLoaders
    train_ds = BananaSegmentationDataset(train_df, base_dir=base_dir, is_training=True)
    val_ds = BananaSegmentationDataset(val_df, base_dir=base_dir, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model, Criterion, Optimizer
    model = UNet(n_channels=3, n_classes=3, base_filters=base_filters).to(device_obj)
    criterion = CombinedDiceCrossEntropyLoss(dice_weight=0.5, ce_weight=0.5).to(device_obj)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_lesion_dice": [],
        "val_lesion_iou": [],
        "val_leaf_dice": [],
        "val_macro_dice": [],
        "best_epoch": 0,
        "best_val_macro_dice": 0.0
    }

    best_val_dice = -1.0
    epochs_no_improve = 0

    print(f"[UNet Train] Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # -------------------------------------------------------------
        # Training Phase
        # -------------------------------------------------------------
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        total_train_batches = len(train_loader)
        for b_idx, (images, masks, _) in enumerate(train_loader, 1):
            images = images.to(device_obj)
            masks = masks.to(device_obj)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

            if b_idx % 10 == 0 or b_idx == total_train_batches:
                print(f"  [Epoch {epoch}/{epochs}] Batch {b_idx}/{total_train_batches} - Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_train_loss = train_loss_accum / max(1, train_batches)

        # -------------------------------------------------------------
        # Validation Phase
        # -------------------------------------------------------------
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device_obj)
                masks = masks.to(device_obj)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss_accum += loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        avg_val_loss = val_loss_accum / max(1, val_batches)

        concat_preds = torch.cat(all_preds, dim=0)
        concat_targets = torch.cat(all_targets, dim=0)
        val_metrics = compute_segmentation_metrics(concat_preds, concat_targets)

        val_lesion_dice = val_metrics["lesion_dice"]
        val_lesion_iou = val_metrics["lesion_iou"]
        val_leaf_dice = val_metrics["leaf_dice"]
        val_macro_dice = val_metrics["macro_dice"]

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_lesion_dice"].append(val_lesion_dice)
        history["val_lesion_iou"].append(val_lesion_iou)
        history["val_leaf_dice"].append(val_leaf_dice)
        history["val_macro_dice"].append(val_macro_dice)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Lesion Dice: {val_lesion_dice:.4f} | "
            f"Macro Dice: {val_macro_dice:.4f}",
            flush=True
        )

        # Save checkpoint if best macro dice achieved
        if val_macro_dice > best_val_dice:
            best_val_dice = val_macro_dice
            epochs_no_improve = 0
            history["best_epoch"] = epoch
            history["best_val_macro_dice"] = val_macro_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "base_filters": base_filters
            }, output_weights_path)
            print(f"  --> Saved new best checkpoint to {output_weights_path} (Macro Dice: {best_val_dice:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[UNet Train] Early stopping triggered: no improvement for {patience} consecutive epochs.", flush=True)
                break

    # Save training history JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[UNet Train] Completed training. History saved to {history_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on Banana Foliar Disease Dataset")
    parser.add_argument("--metadata", type=str, default="data/synthetic/metadata.csv")
    parser.add_argument("--base-dir", type=str, default="data/synthetic")
    parser.add_argument("--weights", type=str, default="results/unet_weights.pt")
    parser.add_argument("--history", type=str, default="results/unet_training_history.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--base-filters", type=int, default=16)
    args = parser.parse_args()

    train_unet(
        metadata_path=args.metadata,
        base_dir=args.base_dir,
        output_weights_path=args.weights,
        history_path=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        base_filters=args.base_filters
    )
