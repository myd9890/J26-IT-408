"""
Evaluation and benchmarking modules for Component 3.
"""

from src.evaluation.expert_simulator import AgroExpertSimulator


def run_all_benchmarks(*args, **kwargs):
    from src.evaluation.run_all_benchmarks import run_all_benchmarks as _run
    return _run(*args, **kwargs)


__all__ = ["AgroExpertSimulator", "run_all_benchmarks"]

