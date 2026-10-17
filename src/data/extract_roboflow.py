"""
Compatibility shim: logic moved to scripts/data/extract_roboflow.py
"""

from scripts.data.extract_roboflow import extract_and_curate, ZIPS, TARGET_SIZE, BASE_DIR, RAW_DIR, PROCESSED_DIR

if __name__ == "__main__":
    extract_and_curate()
