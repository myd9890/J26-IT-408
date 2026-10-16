"""
Evaluation and benchmarking modules for Component 3.
"""

from src.evaluation.expert_simulator import AgroExpertSimulator
from src.evaluation.metrics import compute_segmentation_metrics
from src.evaluation.tables import (
    generate_table1_segmentation_benchmarks,
    generate_table2_severity_benchmarks,
    generate_table3_fusion_benchmarks,
    generate_table4_temporal_benchmarks,
)

# Aliases for convenience
generate_table2_severity_models = generate_table2_severity_benchmarks
generate_table3_fusion_spearman_ablation = generate_table3_fusion_benchmarks
generate_table4_temporal_forecasting = generate_table4_temporal_benchmarks


def run_all_benchmarks(*args, **kwargs):
    from src.evaluation.run_all_benchmarks import run_all_benchmarks as _run
    return _run(*args, **kwargs)


__all__ = [
    "AgroExpertSimulator",
    "compute_segmentation_metrics",
    "generate_table1_segmentation_benchmarks",
    "generate_table2_severity_benchmarks",
    "generate_table3_fusion_benchmarks",
    "generate_table4_temporal_benchmarks",
    "generate_table2_severity_models",
    "generate_table3_fusion_spearman_ablation",
    "generate_table4_temporal_forecasting",
    "run_all_benchmarks",
]
