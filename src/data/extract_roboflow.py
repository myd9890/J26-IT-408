"""
Extract and Curate Roboflow Datasets (BBTV & Yellow Sigatoka) into data/raw and data/processed.
"""

import os
import zipfile
import json
from PIL import Image

BASE_DIR = r"d:\assignments\banana"
RAW_DIR = r"d:\assignments\banana\banana_severity_c3\data\raw"
PROCESSED_DIR = r"d:\assignments\banana\banana_severity_c3\data\processed"
TARGET_SIZE = (256, 256)

ZIPS = [
    {
        "name": "BBTV",
        "zip_path": os.path.join(BASE_DIR, "Banana Bunchy Top Virus.v1i.coco-segmentation.zip"),
        "extract_dir": os.path.join(RAW_DIR, "bbtv"),
        "target_class": "bbtv",
        "prefix": "roboflow_bbtv_",
        "max_images": None  # take all 647 images
    },
    {
        "name": "Yellow_Sigatoka",
        "zip_path": os.path.join(BASE_DIR, "Yellow Sigatoka.v23i.coco.zip"),
        "extract_dir": os.path.join(RAW_DIR, "yellow_sigatoka"),
        "target_class": "yellow_sigatoka",
        "prefix": "roboflow_ysigatoka_",
        "max_images": 1100  # curate ~1100 to balance with 947 Black Sigatoka
    }
]

def extract_and_curate():
    for item in ZIPS:
        zpath = item["zip_path"]
        edir = item["extract_dir"]
        tclass = item["target_class"]
        prefix = item["prefix"]
        max_imgs = item["max_images"]
        
        print(f"\n[{item['name']}] Extracting {zpath} -> {edir} ...")
        os.makedirs(edir, exist_ok=True)
        with zipfile.ZipFile(zpath, 'r') as z:
            z.extractall(edir)
        print(f"[{item['name']}] Extraction complete.")
        
        # Now find all image files extracted
        img_paths = []
        for root, dirs, files in os.walk(edir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_paths.append(os.path.join(root, f))
                    
        print(f"[{item['name']}] Found {len(img_paths)} raw images.")
        if max_imgs and len(img_paths) > max_imgs:
            img_paths = img_paths[:max_imgs]
            print(f"[{item['name']}] Curating {len(img_paths)} images to target directory.")
            
        dst_folder = os.path.join(PROCESSED_DIR, tclass)
        os.makedirs(dst_folder, exist_ok=True)
        
        curated_count = 0
        for i, src_path in enumerate(img_paths):
            base_name = os.path.basename(src_path)
            clean_name = f"{prefix}{i:04d}_{base_name.replace(' ', '_')}"
            if not clean_name.lower().endswith(".jpg"):
                clean_name = os.path.splitext(clean_name)[0] + ".jpg"
                
            dst_path = os.path.join(dst_folder, clean_name)
            try:
                with Image.open(src_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != TARGET_SIZE:
                        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    img.save(dst_path, "JPEG", quality=95)
                curated_count += 1
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                
        print(f"[{item['name']}] Successfully saved {curated_count} images to {dst_folder}")

    # Update summary and catalog
    update_catalog()

def update_catalog():
    print("\nUpdating dataset summary and catalog...")
    catalog = []
    class_counts = {}
    
    for cls in sorted(os.listdir(PROCESSED_DIR)):
        cls_dir = os.path.join(PROCESSED_DIR, cls)
        if os.path.isdir(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[cls] = len(files)
            for f in files:
                rel_path = f"{cls}/{f}"
                catalog.append({
                    "filename": f,
                    "relative_path": rel_path,
                    "disease_label": cls,
                    "processed_dimensions": list(TARGET_SIZE)
                })
                
    summary = {
        "target_resolution": list(TARGET_SIZE),
        "total_images": len(catalog),
        "class_counts": class_counts,
        "sources_used": [
            "BananaLSD (OriginalSet)",
            "PSFD-Musa (Dwarf Cavendish Jahaji & Diseases)",
            "Roboflow BBTV (banana-pest v1)",
            "Roboflow Yellow Sigatoka (sigatoka-uatrp v23)"
        ]
    }
    
    with open(os.path.join(PROCESSED_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(os.path.join(PROCESSED_DIR, "processed_catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)
        
    print("="*60)
    print("FINAL DATASET INVENTORY ACROSS ALL 5 CLASSES")
    print("="*60)
    print(f"Total Processed Images: {len(catalog)}")
    for cls, count in class_counts.items():
        print(f"  - {cls:18s}: {count:5d} images")
    print("="*60)

if __name__ == "__main__":
    extract_and_curate()
