"""
Data Curation and Pre-processing Script (Phase 0 -> T0.6)
Standardizes all extracted raw images into data/processed/ with:
- Unified 256x256 RGB JPEG
- Standardized directory layout matching DiseaseType enums
- Metadata index tracking source attribution, original dimensions, and integrity
"""

import os
import json
from PIL import Image

RAW_DIR = r"d:\assignments\banana\banana_severity_c3\data\raw"
PROCESSED_DIR = r"d:\assignments\banana\banana_severity_c3\data\processed"
TARGET_SIZE = (256, 256)

TARGET_CLASSES = [
    "black_sigatoka",
    "yellow_sigatoka",
    "cordana",
    "healthy"
]

for cls in TARGET_CLASSES:
    os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)

# 1. BananaLSD OriginalSet mapping
bananalsd_base = os.path.join(
    RAW_DIR,
    "bananalsd",
    "Banana Leaf Spot Diseases (BananaLSD) Dataset for Classification of Banana Leaf Diseases Using Machine Learning",
    "BananaLSD",
    "OriginalSet"
)

sources = [
    {
        "name": "BananaLSD",
        "folder": os.path.join(bananalsd_base, "sigatoka"),
        "target_class": "black_sigatoka",
        "prefix": "bananalsd_sigatoka_"
    },
    {
        "name": "BananaLSD",
        "folder": os.path.join(bananalsd_base, "cordana"),
        "target_class": "cordana",
        "prefix": "bananalsd_cordana_"
    },
    {
        "name": "BananaLSD",
        "folder": os.path.join(bananalsd_base, "healthy"),
        "target_class": "healthy",
        "prefix": "bananalsd_healthy_"
    },
    # 2. PSFD-Musa mappings
    {
        "name": "PSFD-Musa",
        "folder": os.path.join(RAW_DIR, "psfd_musa", "extracted", "BANANA DISEASE", "RESIZED BLACK SIGATOKA"),
        "target_class": "black_sigatoka",
        "prefix": "psfd_bsigatoka_"
    },
    {
        "name": "PSFD-Musa",
        "folder": os.path.join(RAW_DIR, "psfd_musa", "extracted", "BANANA DISEASE", "RESIZED YELLOW SIGATOKA"),
        "target_class": "yellow_sigatoka",
        "prefix": "psfd_ysigatoka_"
    },
    {
        "name": "PSFD-Musa",
        "folder": os.path.join(RAW_DIR, "psfd_musa", "extracted", "BANANA VARIETY", "RESIZED JAHAJI LEAF"),
        "target_class": "healthy",
        "prefix": "psfd_jahaji_healthy_"
    },
    {
        "name": "PSFD-Musa",
        "folder": os.path.join(RAW_DIR, "psfd_musa", "extracted", "BANANA VARIETY", "RESIZED MALBHOG LEAF"),
        "target_class": "healthy",
        "prefix": "psfd_malbhog_healthy_",
        "max_images": 400  # Cap to maintain class balance
    }
]

records = []
class_counts = {cls: 0 for cls in TARGET_CLASSES}

print("Starting curation and image normalization...")

for src in sources:
    folder = src["folder"]
    if not os.path.exists(folder):
        print(f"WARNING: Folder not found: {folder}")
        continue
    
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if "max_images" in src:
        files = files[:src["max_images"]]
        
    print(f"Processing {len(files)} images from {src['name']} -> {src['target_class']} ...")
    
    for f in files:
        src_path = os.path.join(folder, f)
        clean_fname = src["prefix"] + f.replace(" ", "_")
        if not clean_fname.lower().endswith(".jpg"):
            clean_fname = os.path.splitext(clean_fname)[0] + ".jpg"
            
        dst_path = os.path.join(PROCESSED_DIR, src["target_class"], clean_fname)
        
        try:
            with Image.open(src_path) as img:
                orig_w, orig_h = img.size
                orig_mode = img.mode
                
                # Convert to RGB if palette/grayscale/RGBA
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                # Resize if not already 256x256
                if img.size != TARGET_SIZE:
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                img.save(dst_path, "JPEG", quality=95)
                
            class_counts[src["target_class"]] += 1
            records.append({
                "filename": clean_fname,
                "relative_path": os.path.relpath(dst_path, PROCESSED_DIR).replace("\\", "/"),
                "disease_label": src["target_class"],
                "source_dataset": src["name"],
                "original_dimensions": [orig_w, orig_h],
                "processed_dimensions": list(TARGET_SIZE)
            })
        except Exception as e:
            print(f"Error processing {src_path}: {e}")

summary = {
    "target_resolution": list(TARGET_SIZE),
    "total_images": len(records),
    "class_counts": class_counts,
    "sources_used": ["BananaLSD (OriginalSet)", "PSFD-Musa (Dwarf Cavendish Jahaji & Diseases)"]
}

summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

index_path = os.path.join(PROCESSED_DIR, "processed_catalog.json")
with open(index_path, "w") as f:
    json.dump(records, f, indent=2)

print("\n" + "="*60)
print("PHASE 0 CURATION AUDIT COMPLETE")
print("="*60)
print(f"Total Processed Images: {len(records)}")
for cls, count in class_counts.items():
    print(f"  - {cls:18s}: {count:5d} images")
print(f"Target Resolution: {TARGET_SIZE} at 95% JPEG quality")
print(f"Catalog saved to: {index_path}")
print(f"Summary saved to: {summary_path}")
