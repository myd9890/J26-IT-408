"""
Public Dataset Adapter & Classical Pseudo-Mask Generator.

Provides:
1. Taxonomy mapping from diverse public folder names to canonical DiseaseType enums.
2. Classical ExG + CIELAB + HSV pseudo-mask generation pipeline for bootstrapping
   segmentation masks when pixel-level annotations are absent.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image

from src.schemas.contracts import DiseaseType


# Taxonomy mapping dictionary from common public folder names to DiseaseType
TAXONOMY_MAP: Dict[str, Tuple[DiseaseType, str]] = {
    "cordana": (DiseaseType.CORDANA, "HIGH"),
    "neocordana": (DiseaseType.CORDANA, "HIGH"),
    "sigatoka": (DiseaseType.BLACK_SIGATOKA, "MEDIUM_HIGH"),
    "black_sigatoka": (DiseaseType.BLACK_SIGATOKA, "HIGH"),
    "black sigatoka": (DiseaseType.BLACK_SIGATOKA, "HIGH"),
    "resized black sigatoka": (DiseaseType.BLACK_SIGATOKA, "HIGH"),
    "yellow_sigatoka": (DiseaseType.YELLOW_SIGATOKA, "HIGH"),
    "yellow sigatoka": (DiseaseType.YELLOW_SIGATOKA, "HIGH"),
    "resized yellow sigatoka": (DiseaseType.YELLOW_SIGATOKA, "HIGH"),
    "bbtv": (DiseaseType.BBTV, "HIGH"),
    "bunchy_top": (DiseaseType.BBTV, "HIGH"),
    "bunchy top": (DiseaseType.BBTV, "HIGH"),
    "banana bunchy top virus": (DiseaseType.BBTV, "HIGH"),
    "healthy": (DiseaseType.HEALTHY, "HIGH"),
    "banana_healthy_leaf": (DiseaseType.HEALTHY, "HIGH"),
    "jahaji_leaf": (DiseaseType.HEALTHY, "HIGH"),
    "resized jahaji leaf": (DiseaseType.HEALTHY, "HIGH"),
    "malbhog_leaf": (DiseaseType.HEALTHY, "HIGH"),
    "resized malbhog leaf": (DiseaseType.HEALTHY, "HIGH"),
}


def map_folder_to_disease_type(folder_name_or_path: str) -> Tuple[DiseaseType, str]:
    """
    Maps an arbitrary public folder string or image path to a canonical DiseaseType enum.

    Args:
        folder_name_or_path: Folder name or full file path string.

    Returns:
        Tuple of (DiseaseType, confidence_string)
    """
    norm = folder_name_or_path.replace("\\", "/").lower()
    base_folder = os.path.basename(os.path.normpath(norm))

    # Exact base match
    if base_folder in TAXONOMY_MAP:
        return TAXONOMY_MAP[base_folder]

    # Keyword heuristic matching across path components
    for keyword, (dtype, conf) in TAXONOMY_MAP.items():
        if keyword in norm:
            return dtype, conf

    # Fallback to healthy if unknown control
    return DiseaseType.HEALTHY, "LOW_FALLBACK"


def generate_pseudo_masks(rgb_image: np.ndarray) -> np.ndarray:
    """
    Generates a 3-class semantic pseudo-mask from an RGB image:
      Class 0: Background
      Class 1: Healthy Leaf Tissue
      Class 2: Necrotic or Chlorotic Lesion Tissue

    Implementation pipeline:
      1. Excess Green Index (ExG = 2G - R - B) + Otsu thresholding + Morphological
         closing to delineate the leaf lamina boundary from background.
      2. Inside the leaf mask, CIELAB (a*, b*) and HSV color-space thresholding
         to segment necrotic lesions (dark brown/black streaks) and chlorotic halos (yellow).

    Args:
        rgb_image: Numpy array of shape (H, W, 3) with dtype uint8 in RGB format.

    Returns:
        mask: Numpy array of shape (H, W) with dtype uint8, containing values in {0, 1, 2}.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {rgb_image.shape}")

    h, w, _ = rgb_image.shape

    # Normalize RGB channels to [0.0, 1.0]
    r = rgb_image[:, :, 0].astype(np.float32) / 255.0
    g = rgb_image[:, :, 1].astype(np.float32) / 255.0
    b = rgb_image[:, :, 2].astype(np.float32) / 255.0

    # -------------------------------------------------------------------------
    # Step 1: Leaf Lamina Segmentation via Excess Green Index (ExG)
    # -------------------------------------------------------------------------
    exg = 2.0 * g - r - b  # Ranges approximately [-2.0, 2.0]

    # Normalize ExG to [0, 255] for Otsu thresholding
    exg_norm = np.clip((exg + 2.0) / 4.0 * 255.0, 0, 255).astype(np.uint8)

    # Otsu automatic thresholding
    _, leaf_mask_raw = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill holes inside the leaf lamina
    kernel_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask_raw, cv2.MORPH_CLOSE, kernel_leaf)
    leaf_bool = leaf_mask > 0

    # Ensure background is clean: find largest connected component if multiple
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    if num_labels > 1:
        # Ignore background label 0, find component with maximum area
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        leaf_bool = (labels == largest_label)

    # -------------------------------------------------------------------------
    # Step 2: Lesion Segmentation inside the Leaf Boundary
    # -------------------------------------------------------------------------
    # Color conversions
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Condition A: Dark necrotic brown/black lesion tissue (Sigatoka / Cordana centers)
    # Characterized by low luminance, shifted a* (reddish-brown), or low brightness
    necrotic_condition = (
        leaf_bool &
        (
            ((a_channel >= 125) & (l_channel < 135)) |
            ((h_channel <= 25) & (v_channel < 130) & (s_channel > 50)) |
            ((l_channel < 60) & (v_channel < 75))
        )
    )

    # Condition B: Chlorotic yellow halos / marginal chlorosis (Yellow Sigatoka / Cordana / BBTV)
    # Characterized by yellow-brown hue distinctly lower than healthy green (H < 34) and elevated a*
    chlorotic_condition = (
        leaf_bool &
        (
            (h_channel >= 15) & (h_channel <= 32) & (s_channel > 60) & (a_channel >= 114)
        )
    )

    # Combined raw lesion mask
    lesion_raw = (necrotic_condition | chlorotic_condition).astype(np.uint8) * 255

    # Filter isolated pixel noise (< 5 pixels) using connected components
    num_les_labels, les_labels, les_stats, _ = cv2.connectedComponentsWithStats(lesion_raw, connectivity=8)
    lesion_clean = np.zeros_like(lesion_raw)
    min_lesion_pixels = 5
    for lbl in range(1, num_les_labels):
        if les_stats[lbl, cv2.CC_STAT_AREA] >= min_lesion_pixels:
            lesion_clean[les_labels == lbl] = 255

    lesion_bool = (lesion_clean > 0) & leaf_bool

    # -------------------------------------------------------------------------
    # Step 3: Construct 3-Class Output Mask
    # -------------------------------------------------------------------------
    mask_3class = np.zeros((h, w), dtype=np.uint8)
    mask_3class[leaf_bool] = 1   # Healthy leaf tissue
    mask_3class[lesion_bool] = 2 # Diseased lesion tissue

    return mask_3class


def load_image_and_generate_mask(image_path: str, target_size: Optional[Tuple[int, int]] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image from disk, resizes to target_size, and returns both the RGB array
    and the corresponding 3-class pseudo-mask.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        if target_size is not None and img_rgb.size != target_size:
            img_rgb = img_rgb.resize(target_size, Image.Resampling.LANCZOS)
        rgb_np = np.array(img_rgb, dtype=np.uint8)

    mask = generate_pseudo_masks(rgb_np)
    return rgb_np, mask
