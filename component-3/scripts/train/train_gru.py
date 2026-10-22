"""
Standalone CLI entry point for PyTorch GRU Progression Forecaster Training.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.training.gru_trainer import train_gru

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU on Longitudinal Progression Dataset")
    parser.add_argument("--metadata", type=str, default="data/synthetic/metadata.csv")
    parser.add_argument("--weights", type=str, default="results/gru_forecaster.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--hidden-size", type=int, default=32)
    args = parser.parse_args()

    train_gru(
        metadata_path=args.metadata,
        output_weights_path=args.weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size
    )
