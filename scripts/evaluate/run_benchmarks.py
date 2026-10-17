"""
Standalone CLI Benchmark Suite Runner for Component 3.

Produces:
- Tables 1-4 in results/tables/
- Figures 1-4 in results/figures/
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.dataset import plant_level_split
from src.evaluation.tables import (
    generate_table1_segmentation_benchmarks,
    generate_table2_severity_benchmarks,
    generate_table3_fusion_benchmarks,
    generate_table4_temporal_benchmarks,
)
from scripts.evaluate.generate_figures import (
    generate_figure1_segmentation_comparison,
    generate_figure2_confusion_matrix,
    generate_figure3_zero_collapse,
    generate_figure4_trajectories,
)


def main():
    print("[Benchmarks] Loading longitudinal cohort metadata...")
    meta_df = pd.read_csv("data/synthetic/metadata.csv")
    train_df, val_df, test_df = plant_level_split(meta_df, 0.70, 0.15, 0.15, seed=42)

    # Tables 1 to 4
    generate_table1_segmentation_benchmarks(test_df)
    generate_table2_severity_benchmarks(train_df, test_df)
    generate_table3_fusion_benchmarks(test_df)
    generate_table4_temporal_benchmarks(test_df)

    # Figures 1 to 4
    generate_figure1_segmentation_comparison()
    generate_figure2_confusion_matrix(test_df)
    generate_figure3_zero_collapse()
    generate_figure4_trajectories()

    print("[SUCCESS] All 4 thesis benchmark tables and 4 figures generated successfully!")


if __name__ == "__main__":
    main()
