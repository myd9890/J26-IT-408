"""
Dataset Splitter & PyTorch Segmentation Dataset for Banana Foliar Disease.

Implements:
1. Strict plant-level 70/15/15 partitioning (Section 3.8) to eliminate longitudinal
   data leakage across training, validation, and test sets.
2. PyTorch BananaSegmentationDataset returning (image_tensor, mask_tensor, metadata_dict).
3. Spatial data augmentations (flips, rotations) while preserving photic color fidelity.
"""

import os
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def plant_level_split(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits longitudinal inspection data strictly at the plant level (plant_id),
    guaranteeing that no observations from the same plant appear across multiple splits.

    Args:
        metadata_df: DataFrame containing at least a 'plant_id' column.
        train_ratio: Proportion of unique plants for training (default 0.70).
        val_ratio: Proportion of unique plants for validation (default 0.15).
        test_ratio: Proportion of unique plants for testing (default 0.15).
        seed: Random seed for reproducible splitting.

    Returns:
        (train_df, val_df, test_df)
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    unique_plants = sorted(metadata_df["plant_id"].unique())
    num_plants = len(unique_plants)

    rng = np.random.default_rng(seed)
    shuffled_plants = rng.permutation(unique_plants)

    num_train = int(round(num_plants * train_ratio))
    num_val = int(round(num_plants * val_ratio))
    num_test = num_plants - num_train - num_val

    train_plants = set(shuffled_plants[:num_train])
    val_plants = set(shuffled_plants[num_train : num_train + num_val])
    test_plants = set(shuffled_plants[num_train + num_val :])

    # Assert strict disjoint sets (0% leakage)
    assert len(train_plants.intersection(val_plants)) == 0, "Train and Val plant sets overlap!"
    assert len(train_plants.intersection(test_plants)) == 0, "Train and Test plant sets overlap!"
    assert len(val_plants.intersection(test_plants)) == 0, "Val and Test plant sets overlap!"

    train_df = metadata_df[metadata_df["plant_id"].isin(train_plants)].copy().reset_index(drop=True)
    val_df = metadata_df[metadata_df["plant_id"].isin(val_plants)].copy().reset_index(drop=True)
    test_df = metadata_df[metadata_df["plant_id"].isin(test_plants)].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


class BananaSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Banana Leaf & Lesion Semantic Segmentation.
    Yields:
      - image_tensor: torch.FloatTensor of shape (3, H, W) normalized to [0, 1].
      - mask_tensor: torch.LongTensor of shape (H, W) with class values in {0, 1, 2}.
      - metadata: dict of record annotations.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        base_dir: str,
        is_training: bool = False,
        target_size: Tuple[int, int] = (256, 256)
    ):
        """
        Args:
            metadata_df: DataFrame with image_path, mask_path, and disease metadata.
            base_dir: Root directory where image_path and mask_path are referenced.
            is_training: Whether to apply stochastic spatial data augmentation.
            target_size: (H, W) target tensor resolution.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.is_training = is_training
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]

        # Resolve paths
        img_rel = row["image_path"]
        mask_rel = row["mask_path"]

        img_full = os.path.join(self.base_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
        mask_full = os.path.join(self.base_dir, mask_rel) if not os.path.isabs(mask_rel) else mask_rel

        # Load RGB image and mask
        with Image.open(img_full) as img:
            img = img.convert("RGB")
            if img.size != (self.target_size[1], self.target_size[0]):
                img = img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)
            img_tensor = TF.to_tensor(img)  # Shape (3, H, W) in [0.0, 1.0]

        with Image.open(mask_full) as msk:
            if msk.size != (self.target_size[1], self.target_size[0]):
                msk = msk.resize((self.target_size[1], self.target_size[0]), Image.Resampling.NEAREST)
            mask_np = np.array(msk, dtype=np.int64)
            # Ensure mask values are in {0, 1, 2}
            mask_np = np.clip(mask_np, 0, 2)
            mask_tensor = torch.from_numpy(mask_np)  # Shape (H, W)

        # Apply spatial-only data augmentations during training
        if self.is_training:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            # Random 90-degree rotation
            if torch.rand(1).item() > 0.5:
                rot_k = int(torch.randint(1, 4, (1,)).item())
                img_tensor = torch.rot90(img_tensor, k=rot_k, dims=[1, 2])
                mask_tensor = torch.rot90(mask_tensor, k=rot_k, dims=[0, 1])

        meta_dict = row.to_dict()
        return img_tensor, mask_tensor, meta_dict
