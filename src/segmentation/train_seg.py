"""
U-Net Training and Evaluation Routine for Banana Foliar Disease Segmentation.

(Compatibility Facade - logic moved to src.training.seg_trainer)
"""

import argparse
from src.training.seg_trainer import train_unet

__all__ = ["train_unet"]

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
