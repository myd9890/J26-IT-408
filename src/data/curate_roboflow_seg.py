"""
Compatibility shim: logic moved to scripts/data/curate_roboflow_seg.py
"""

from scripts.data.curate_roboflow_seg import (
    CATEGORY_MAP,
    TARGET_SIZE,
    rasterize_coco_split,
    run_curation,
)

if __name__ == "__main__":
    run_curation()
