"""
Morphometric Feature Extraction Engine for Banana Foliar Disease Assessment.

Extracts multidimensional foliar geometric and spatial features from 3-class segmentation masks:
- Scope %: Foliar disease coverage (Lesion Pixels / Leaf Pixels * 100)
- Lesion Count: Number of distinct contiguous lesion components (filtered for noise >= 5px)
- Coalescence Ratio % (R_coal): Area of the largest single lesion relative to total leaf area
- Leaf Aspect Ratio (AR_leaf): Minimum-area oriented bounding box aspect ratio
- Lamina Distortion Index (D_morph): Stunting and bunching proxy relative to healthy Cavendish (3.5)
"""

from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
import cv2
import torch


@dataclass
class MorphometricFeatures:
    """
    Structured container for extracted foliar morphometrics.
    All percentage features are bounded in [0.0, 100.0].
    """
    scope_pct: float
    num_lesions: int
    coalescence_ratio_pct: float
    leaf_aspect_ratio: float
    morph_distortion_idx: float
    leaf_pixels: int
    lesion_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a standard dictionary."""
        return asdict(self)


def extract_leaf_and_lesion_features(
    mask: Union[np.ndarray, torch.Tensor],
    min_lesion_pixels: int = 5,
    reference_aspect_ratio: float = 3.5,
    min_aspect_ratio: float = 1.5
) -> MorphometricFeatures:
    """
    Extracts morphometric and foliar lesion coverage features from a 3-class mask.

    Classes expected:
      0: Background
      1: Healthy leaf tissue
      2: Lesion tissue

    Args:
        mask: 2D array or tensor of shape (H, W) with pixel classes {0, 1, 2}.
        min_lesion_pixels: Area threshold in pixels to filter out noise artifacts.
        reference_aspect_ratio: Standard healthy Cavendish leaf length-to-width ratio (~3.5).
        min_aspect_ratio: Severely stunted/bunched Cavendish ratio (~1.5).

    Returns:
        MorphometricFeatures instance with bounded numerical metrics.
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_np = np.asarray(mask, dtype=np.uint8)

    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask of shape (H, W), got {mask_np.shape}")

    # Total leaf lamina includes both healthy tissue (1) and lesion tissue (2)
    leaf_bool = (mask_np == 1) | (mask_np == 2)
    lesion_bool = (mask_np == 2)

    leaf_pixels = int(np.count_nonzero(leaf_bool))
    lesion_pixels = int(np.count_nonzero(lesion_bool))

    # -------------------------------------------------------------------------
    # 1. Foliar Scope Percentage
    # -------------------------------------------------------------------------
    if leaf_pixels == 0:
        scope_pct = 0.0
    else:
        scope_pct = (lesion_pixels / leaf_pixels) * 100.0
    scope_pct = float(np.clip(scope_pct, 0.0, 100.0))

    # -------------------------------------------------------------------------
    # 2. Lesion Count & Coalescence Ratio via Connected Components
    # -------------------------------------------------------------------------
    num_lesions = 0
    max_lesion_area = 0

    if lesion_pixels > 0:
        lesion_uint8 = lesion_bool.astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lesion_uint8, connectivity=8)

        # Label 0 is background in connectedComponents
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= min_lesion_pixels:
                num_lesions += 1
                if area > max_lesion_area:
                    max_lesion_area = area

    if leaf_pixels == 0:
        coalescence_ratio_pct = 0.0
    else:
        coalescence_ratio_pct = (max_lesion_area / leaf_pixels) * 100.0
    coalescence_ratio_pct = float(np.clip(coalescence_ratio_pct, 0.0, 100.0))

    # -------------------------------------------------------------------------
    # 3. Leaf Aspect Ratio & Morphological Distortion Index
    # -------------------------------------------------------------------------
    if leaf_pixels > 20:
        leaf_uint8 = leaf_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(leaf_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 10:
                rect = cv2.minAreaRect(largest_contour)
                (_, (dim1, dim2), _) = rect
                major_axis = max(dim1, dim2)
                minor_axis = max(1.0, min(dim1, dim2))
                leaf_aspect_ratio = float(major_axis / minor_axis)
            else:
                leaf_aspect_ratio = reference_aspect_ratio
        else:
            leaf_aspect_ratio = reference_aspect_ratio
    else:
        # Fallback to reference healthy aspect ratio if no leaf detected
        leaf_aspect_ratio = reference_aspect_ratio

    # Bounded distortion index: 0.0 = healthy slender lamina, 100.0 = severe bunching / stunting
    # Formula: D_morph = clip((AR_ref - AR_leaf) / (AR_ref - AR_min) * 100.0, 0.0, 100.0)
    denom = reference_aspect_ratio - min_aspect_ratio
    if denom <= 0:
        morph_distortion_idx = 0.0
    else:
        raw_distortion = ((reference_aspect_ratio - leaf_aspect_ratio) / denom) * 100.0
        morph_distortion_idx = float(np.clip(raw_distortion, 0.0, 100.0))

    return MorphometricFeatures(
        scope_pct=round(scope_pct, 2),
        num_lesions=int(num_lesions),
        coalescence_ratio_pct=round(coalescence_ratio_pct, 2),
        leaf_aspect_ratio=round(leaf_aspect_ratio, 2),
        morph_distortion_idx=round(morph_distortion_idx, 2),
        leaf_pixels=leaf_pixels,
        lesion_pixels=lesion_pixels
    )
