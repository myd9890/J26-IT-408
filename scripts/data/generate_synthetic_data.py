"""
Standalone CLI entry point for Longitudinal Synthetic Banana Cohort Generation.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic_generator import generate_synthetic_cohort

if __name__ == "__main__":
    generate_synthetic_cohort()
