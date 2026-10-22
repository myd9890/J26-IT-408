"""
Compatibility shim: logic moved to scripts/data/curate_raw_to_processed.py
"""

from scripts.data.curate_raw_to_processed import (
    TARGET_SIZE,
    TARGET_CLASSES,
    run_curation,
)

if __name__ == "__main__":
    run_curation()
