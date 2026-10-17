"""
Standalone CLI entry point for XGBoost Severity Fusion Training.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.training.xgb_trainer import train_xgboost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on Multimodal Foliar Disease Dataset")
    parser.add_argument("--metadata", type=str, default="data/synthetic/metadata.csv")
    parser.add_argument("--output", type=str, default="results/xgboost_model.json")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    train_xgboost(
        metadata_path=args.metadata,
        output_model_path=args.output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
