"""
Curate and Rasterize Roboflow Banana Leaf Disease 2 Instance Segmentation Dataset.

Processes COCO-segmentation polygons into:
- 256x256 RGB images in data/real_seg/images/
- 256x256 3-class ground-truth masks in data/real_seg/masks/:
    0: Background
    1: Healthy leaf lamina
    2: Diseased foliar lesion (Sigatoka / Cordana / Pestalotiopsis)
- Comprehensive metadata index in data/real_seg/metadata.csv with real foliar scope %.
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image

RAW_DIR = r"data/raw/banana_leaf_disease_2"
OUTPUT_DIR = r"data/real_seg"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
TARGET_SIZE = (256, 256)

CATEGORY_MAP = {
    "Sigatoka": "black_sigatoka",
    "Cordana": "cordana",
    "Healty": "healthy",
    "Pestalotiopsis": "cordana",  # Map other foliar leaf spots to leaf spot category
    "Leaf": "healthy"
}

def rasterize_coco_split(split_name: str, raw_dir: str):
    split_dir = os.path.join(raw_dir, split_name)
    ann_file = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return []

    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    
    # Group annotations by image_id
    img_id_to_anns = {}
    for a in coco.get("annotations", []):
        img_id_to_anns.setdefault(a["image_id"], []).append(a)

    records = []

    print(f"[{split_name.upper()}] Processing {len(coco.get('images', []))} images...")

    for img_info in coco.get("images", []):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        orig_w = img_info["width"]
        orig_h = img_info["height"]

        img_path = os.path.join(split_dir, fname)
        if not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as im:
                rgb_orig = np.array(im.convert("RGB"))
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        # 1. Build original resolution 3-class mask
        mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Pixels with non-trivial RGB values belong to the leaf lamina (exclude black augmentation padding)
        is_leaf_blade = (rgb_orig.sum(axis=2) > 15)
        mask_orig[is_leaf_blade] = 1  # Default healthy leaf tissue

        anns = img_id_to_anns.get(img_id, [])
        primary_disease = "healthy"
        num_lesions = 0

        for ann in anns:
            cname = cat_id_to_name.get(ann["category_id"], "Leaf")
            canonical_cat = CATEGORY_MAP.get(cname, "healthy")
            
            # If any lesion category is present, adopt it
            if canonical_cat in ["black_sigatoka", "cordana"]:
                primary_disease = canonical_cat

            # Rasterize lesion polygons as Class 2
            if cname in ["Sigatoka", "Cordana", "Pestalotiopsis"]:
                for seg in ann.get("segmentation", []):
                    if len(seg) >= 6:  # At least 3 (x, y) coordinate pairs
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask_orig, [poly], 2)
                        num_lesions += 1

        # In pure healthy images, ensure lesions remain 0
        if primary_disease == "healthy" and num_lesions == 0:
            mask_orig[mask_orig == 2] = 1

        # 2. Resize both Image and Mask to unified 256x256
        rgb_resized = cv2.resize(rgb_orig, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_orig, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # Clean filename for saving
        clean_stem = f"real_{split_name}_{img_id:04d}_{os.path.splitext(fname)[0]}"
        clean_img_name = clean_stem + ".jpg"
        clean_mask_name = clean_stem + ".png"

        out_img_path = os.path.join(IMAGES_DIR, clean_img_name)
        out_mask_path = os.path.join(MASKS_DIR, clean_mask_name)

        Image.fromarray(rgb_resized).save(out_img_path, quality=95)
        Image.fromarray(mask_resized).save(out_mask_path)

        # 3. Compute ground truth foliar metrics
        leaf_px = int((mask_resized >= 1).sum())
        lesion_px = int((mask_resized == 2).sum())
        scope_pct = round(float(lesion_px / max(1, leaf_px) * 100.0), 2)

        # Generate plant_id to preserve plant-level partitioning
        plant_id = f"REAL_PLANT_{split_name[:2].upper()}_{(img_id % 60):03d}"

        records.append({
            "image_path": os.path.join("images", clean_img_name).replace("\\", "/"),
            "mask_path": os.path.join("masks", clean_mask_name).replace("\\", "/"),
            "plant_id": plant_id,
            "disease_label": primary_disease,
            "ground_truth_scope_pct": scope_pct,
            "num_lesions": num_lesions,
            "split": "val" if split_name == "valid" else split_name,
            "orig_width": orig_w,
            "orig_height": orig_h
        })

    return records


def run_curation():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

    all_records = []
    for s in ["train", "valid", "test"]:
        recs = rasterize_coco_split(s, RAW_DIR)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    metadata_csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    df.to_csv(metadata_csv_path, index=False)

    print("\n" + "=" * 60)
    print("REAL SEGMENTATION CURATION COMPLETE")
    print("=" * 60)
    print(f"Total Images Curated: {len(df)}")
    print("Class distribution:")
    print(df["disease_label"].value_counts().to_string())
    print("\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print(f"\nAverage Foliar Scope %: {df['ground_truth_scope_pct'].mean():.2f}%")
    print(f"Max Scope %: {df['ground_truth_scope_pct'].max():.2f}%")
    print(f"Metadata saved to: {metadata_csv_path}")


if __name__ == "__main__":
    run_curation()
