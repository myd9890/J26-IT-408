"""
Standalone CLI entry point for U-Net Semantic Segmentation Training.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.training.seg_trainer import train_unet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on Banana Foliar Disease Dataset")
    parser.add_argument("--metadata", type=str, default="data/synthetic/metadata.csv")
    parser.add_argument("--base-dir", type=str, default="data/synthetic")
    parser.add_argument("--weights", type=str, default="results/unet_weights.pt")
    parser.add_argument("--history", type=str, default="results/unet_training_history.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--base-filters", type=int, default=32)
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
