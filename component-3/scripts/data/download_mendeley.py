"""
Download and extract Mendeley datasets for Phase 0:
1. BananaLSD (9tb7k297ff) -> data/raw/bananalsd/
2. PSFD-Musa (4wyymrcpyz) -> data/raw/psfd_musa/
"""

import os
import sys
import urllib.request
import zipfile
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DATASETS = [
    {
        "name": "BananaLSD",
        "id": "9tb7k297ff",
        "zip_path": os.path.join(RAW_DIR, "bananalsd.zip"),
        "extract_dir": os.path.join(RAW_DIR, "bananalsd")
    },
    {
        "name": "PSFD-Musa",
        "id": "4wyymrcpyz",
        "zip_path": os.path.join(RAW_DIR, "psfd_musa.zip"),
        "extract_dir": os.path.join(RAW_DIR, "psfd_musa")
    }
]


def download_file(url, target_path, name):
    print(f"\n[{name}] Initiating download from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    
    start_time = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, 'wb') as out_f:
        total_size = int(resp.headers.get('content-length', 0))
        print(f"[{name}] Total size: {total_size / (1024*1024):.2f} MB")
        
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB blocks
        while True:
            buffer = resp.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_f.write(buffer)
            pct = (downloaded / total_size * 100) if total_size > 0 else 0
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r[{name}] Progress: {mb:.1f} MB ({pct:.1f}%)")
            sys.stdout.flush()
            
    elapsed = time.time() - start_time
    print(f"\n[{name}] Download finished in {elapsed:.1f}s ({downloaded / (1024*1024):.2f} MB)")


def extract_zip(zip_path, extract_dir, name):
    print(f"\n[{name}] Extracting {zip_path} -> {extract_dir} ...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print(f"[{name}] Extraction complete.")


def main():
    for d in DATASETS:
        url = f"https://prod-dload-1fd9-or-files.s3.amazonaws.com/{d['id']}/1"
        if not os.path.exists(d["zip_path"]):
            download_file(url, d["zip_path"], d["name"])
        extract_zip(d["zip_path"], d["extract_dir"], d["name"])


if __name__ == "__main__":
    main()
