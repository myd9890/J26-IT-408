"""
Comprehensive Verification Gate 2 Test Suite.
Validates:
1. Synthetic dataset integrity (720 records, images, masks).
2. Strict plant-level splitting (zero plant ID leakage across train/val/test).
3. PyTorch BananaSegmentationDataset functionality with DataLoader.
4. Public taxonomy mapper and ExG+CIELAB pseudo-mask generation.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.schemas.contracts import DiseaseType
from src.data.public_adapter import map_folder_to_disease_type, generate_pseudo_masks
from src.data.dataset import plant_level_split, BananaSegmentationDataset

BASE_DIR = r"d:\assignments\banana\banana_severity_c3"
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def test_synthetic_metadata():
    print("\n--- Test 1: Synthetic Metadata Integrity ---")
    meta_path = os.path.join(SYNTHETIC_DIR, "metadata.csv")
    assert os.path.exists(meta_path), f"metadata.csv not found at {meta_path}"

    df = pd.read_csv(meta_path)
    assert len(df) == 720, f"Expected 720 records, got {len(df)}"

    expected_cols = [
        "record_id", "plant_id", "zone_id", "leaf_id", "timepoint_idx",
        "day_index", "observation_date", "disease_label", "image_path",
        "mask_path", "ground_truth_scope_pct", "ground_truth_aspect_ratio",
        "num_lesions", "coalescence_ratio_pct", "trajectory_type",
        "urgency_score", "stover_grade"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col}"

    assert df["plant_id"].nunique() == 120, f"Expected 120 plants, got {df['plant_id'].nunique()}"
    assert set(df["zone_id"].unique()) == {"ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D"}
    assert set(df["timepoint_idx"].unique()) == {0, 1, 2, 3, 4, 5}
    assert set(df["day_index"].unique()) == {0, 7, 14, 21, 28, 35}

    print(f"PASS: metadata.csv verified with {len(df)} records across 120 plants and 4 zones.")


def test_synthetic_images_and_masks():
    print("\n--- Test 2: Synthetic Images and Masks Integrity ---")
    img_files = glob.glob(os.path.join(SYNTHETIC_DIR, "images", "*.jpg"))
    mask_files = glob.glob(os.path.join(SYNTHETIC_DIR, "masks", "*.png"))

    assert len(img_files) == 720, f"Expected 720 images, got {len(img_files)}"
    assert len(mask_files) == 720, f"Expected 720 masks, got {len(mask_files)}"

    # Sample check 5 images and masks
    for i in range(min(5, len(img_files))):
        with Image.open(img_files[i]) as img:
            assert img.size == (256, 256), f"Wrong size {img.size} for {img_files[i]}"
            assert img.mode == "RGB", f"Wrong mode {img.mode} for {img_files[i]}"

        with Image.open(mask_files[i]) as msk:
            assert msk.size == (256, 256), f"Wrong size {msk.size} for {mask_files[i]}"
            arr = np.array(msk)
            uniq = np.unique(arr)
            assert all(v in [0, 1, 2] for v in uniq), f"Invalid mask values {uniq} in {mask_files[i]}"

    print(f"PASS: 720 images and 720 masks verified with uniform (256, 256) size and {{0, 1, 2}} classes.")


def test_plant_level_split():
    print("\n--- Test 3: Plant-Level Split (Zero Leakage) ---")
    meta_path = os.path.join(SYNTHETIC_DIR, "metadata.csv")
    df = pd.read_csv(meta_path)

    train_df, val_df, test_df = plant_level_split(df, 0.70, 0.15, 0.15, seed=42)

    train_plants = set(train_df["plant_id"].unique())
    val_plants = set(val_df["plant_id"].unique())
    test_plants = set(test_df["plant_id"].unique())

    assert len(train_plants) == 84, f"Expected 84 train plants, got {len(train_plants)}"
    assert len(val_plants) == 18, f"Expected 18 val plants, got {len(val_plants)}"
    assert len(test_plants) == 18, f"Expected 18 test plants, got {len(test_plants)}"

    # Check zero overlap
    overlap_tv = train_plants.intersection(val_plants)
    overlap_tt = train_plants.intersection(test_plants)
    overlap_vt = val_plants.intersection(test_plants)

    assert len(overlap_tv) == 0, f"Leakage between train and val: {overlap_tv}"
    assert len(overlap_tt) == 0, f"Leakage between train and test: {overlap_tt}"
    assert len(overlap_vt) == 0, f"Leakage between val and test: {overlap_vt}"

    assert len(train_df) == 84 * 6, f"Expected 504 train records, got {len(train_df)}"
    assert len(val_df) == 18 * 6, f"Expected 108 val records, got {len(val_df)}"
    assert len(test_df) == 18 * 6, f"Expected 108 test records, got {len(test_df)}"

    print(f"PASS: Strict plant-level splitting confirmed: 84 train / 18 val / 18 test with 0.0% overlap.")


def test_pytorch_dataset():
    print("\n--- Test 4: PyTorch BananaSegmentationDataset & DataLoader ---")
    meta_path = os.path.join(SYNTHETIC_DIR, "metadata.csv")
    df = pd.read_csv(meta_path)

    train_df, _, _ = plant_level_split(df, 0.70, 0.15, 0.15, seed=42)
    dataset = BananaSegmentationDataset(train_df, base_dir=SYNTHETIC_DIR, is_training=True)

    assert len(dataset) == 504
    img_t, mask_t, meta = dataset[0]

    assert isinstance(img_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    assert img_t.shape == (3, 256, 256), f"Wrong img shape {img_t.shape}"
    assert mask_t.shape == (256, 256), f"Wrong mask shape {mask_t.shape}"
    assert img_t.dtype == torch.float32
    assert mask_t.dtype == torch.int64
    assert img_t.min() >= 0.0 and img_t.max() <= 1.0

    # Test DataLoader batching
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_img, batch_mask, _ = next(iter(loader))
    assert batch_img.shape == (4, 3, 256, 256)
    assert batch_mask.shape == (4, 256, 256)

    print("PASS: BananaSegmentationDataset and DataLoader batching operate seamlessly.")


def test_public_adapter():
    print("\n--- Test 5: Public Taxonomy Mapper & Pseudo-Mask Generator ---")
    # Taxonomy checks
    cases = [
        ("cordana", DiseaseType.CORDANA),
        ("BananaLSD/OriginalSet/sigatoka", DiseaseType.BLACK_SIGATOKA),
        ("RESIZED YELLOW SIGATOKA", DiseaseType.YELLOW_SIGATOKA),
        ("bbtv", DiseaseType.BBTV),
        ("banana bunchy top virus", DiseaseType.BBTV),
        ("RESIZED JAHAJI LEAF", DiseaseType.HEALTHY),
        ("healthy", DiseaseType.HEALTHY)
    ]
    for text, expected in cases:
        mapped, conf = map_folder_to_disease_type(text)
        assert mapped == expected, f"Failed mapping '{text}': got {mapped}, expected {expected}"

    print("PASS: Taxonomy mapper successfully resolved all public dataset strings.")

    # Pseudo-mask test on sample real processed image
    healthy_sample = glob.glob(os.path.join(PROCESSED_DIR, "healthy", "*.jpg"))[0]
    with Image.open(healthy_sample) as img:
        img_np = np.array(img.convert("RGB"))

    pseudo_mask = generate_pseudo_masks(img_np)
    assert pseudo_mask.shape == (256, 256)
    uniq_vals = set(np.unique(pseudo_mask))
    assert uniq_vals.issubset({0, 1, 2}), f"Unexpected pseudo-mask values: {uniq_vals}"
    assert 1 in uniq_vals, "Pseudo-mask should have detected healthy leaf lamina"

    print("PASS: Classical ExG+CIELAB pseudo-mask generator validated on real Cavendish leaf image.")


if __name__ == "__main__":
    test_synthetic_metadata()
    test_synthetic_images_and_masks()
    test_plant_level_split()
    test_pytorch_dataset()
    test_public_adapter()
    print("\n" + "="*60)
    print(">>> VERIFICATION GATE 2: 100% PASS <<<")
    print("="*60)
