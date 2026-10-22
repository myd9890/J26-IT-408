"""
Automated Thesis Benchmark & Publication Figure Generator (T6.4).

(Modularized facade orchestrating src.evaluation.tables and scripts.evaluate.generate_figures)
"""

import pandas as pd
from src.data.dataset import plant_level_split
from src.evaluation.tables import (
    generate_table1_segmentation_benchmarks,
    generate_table2_severity_benchmarks,
    generate_table3_fusion_benchmarks,
    generate_table4_temporal_benchmarks,
    TABLES_DIR,
)
from scripts.evaluate.generate_figures import (
    generate_figure1_segmentation_comparison,
    generate_figure2_confusion_matrix,
    generate_figure3_zero_collapse,
    generate_figure4_trajectories,
    FIGURES_DIR,
)

__all__ = [
    "generate_table1_segmentation_benchmarks",
    "generate_table2_severity_benchmarks",
    "generate_table3_fusion_benchmarks",
    "generate_table4_temporal_benchmarks",
    "generate_figure1_segmentation_comparison",
    "generate_figure2_confusion_matrix",
    "generate_figure3_zero_collapse",
    "generate_figure4_trajectories",
    "run_all_benchmarks",
    "TABLES_DIR",
    "FIGURES_DIR",
]


def run_all_benchmarks():
    """Executes full suite producing Tables 1-4 and Figures 1-4."""
    meta_df = pd.read_csv("data/synthetic/metadata.csv")
    train_df, val_df, test_df = plant_level_split(meta_df, 0.70, 0.15, 0.15, seed=42)

    generate_table1_segmentation_benchmarks(test_df)
    generate_table2_severity_benchmarks(train_df, test_df)
    generate_table3_fusion_benchmarks(test_df)
    generate_table4_temporal_benchmarks(test_df)

    generate_figure1_segmentation_comparison()
    generate_figure2_confusion_matrix(test_df)
    generate_figure3_zero_collapse()
    generate_figure4_trajectories()

    print("[SUCCESS] All 4 thesis benchmark tables and 4 figures generated successfully!")


if __name__ == "__main__":
    run_all_benchmarks()
