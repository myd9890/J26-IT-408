"""
Classical Segmentation Baseline for Banana Foliar Disease.

Implements ClassicalSegmenter using an established rule-based computer vision pipeline:
1. Leaf lamina segmentation via Excess Green Index (ExG = 2G - R - B) + Otsu thresholding
   + morphological closing + largest connected component extraction.
2. Foliar lesion segmentation via dual-color space thresholding:
   - CIELAB (L*, a*, b*): Reddish-brown necrotic tissue (elevated a*, low L*).
   - HSV (H, S, V): Chlorotic yellow halos and marginal yellowing (H in [15, 32], elevated S).
3. Connected component area filtering (removes single-pixel sensor noise < 5 pixels).
4. Produces a standard 3-class segmentation mask:
   - 0: Background
   - 1: Healthy leaf tissue
   - 2: Lesion tissue

Serves as the classical benchmark baseline against the deep learning U-Net architecture.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import cv2
from PIL import Image


class ClassicalSegmenter:
    """
    Classical rule-based segmenter for banana leaf and lesion segmentation
    using ExG, Otsu thresholding, CIELAB, and HSV color analysis.
    """

    def __init__(
        self,
        min_lesion_pixels: int = 5,
        target_size: Optional[Tuple[int, int]] = (256, 256)
    ):
        """
        Args:
            min_lesion_pixels: Minimum contiguous area in pixels for a valid lesion.
            target_size: Optional (height, width) to resize images during inference.
        """
        self.min_lesion_pixels = min_lesion_pixels
        self.target_size = target_size

    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Predicts a 3-class segmentation mask from an RGB image.

        Args:
            rgb_image: uint8 numpy array of shape (H, W, 3) in RGB order.

        Returns:
            uint8 numpy array of shape (H, W) with pixel classes:
                0 = Background
                1 = Healthy leaf tissue
                2 = Lesion tissue
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3), got {rgb_image.shape}")

        if self.target_size is not None:
            th, tw = self.target_size
            if (rgb_image.shape[0], rgb_image.shape[1]) != (th, tw):
                rgb_image = cv2.resize(rgb_image, (tw, th), interpolation=cv2.INTER_AREA)

        h, w, _ = rgb_image.shape

        # Normalize RGB to [0.0, 1.0] for ExG calculation
        r = rgb_image[:, :, 0].astype(np.float32) / 255.0
        g = rgb_image[:, :, 1].astype(np.float32) / 255.0
        b = rgb_image[:, :, 2].astype(np.float32) / 255.0

        # ---------------------------------------------------------------------
        # Step 1: Leaf Lamina Segmentation via Excess Green Index (ExG)
        # ---------------------------------------------------------------------
        exg = 2.0 * g - r - b  # Approx range [-2.0, 2.0]
        exg_norm = np.clip((exg + 2.0) / 4.0 * 255.0, 0, 255).astype(np.uint8)

        # Otsu thresholding
        _, leaf_mask_raw = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to seal internal venation gaps
        kernel_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask_raw, cv2.MORPH_CLOSE, kernel_leaf)

        # Retain largest connected component as the leaf lamina
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            leaf_raw_bool = (labels == largest_label)

            # Fill internal holes inside the outer leaf contour to enclose necrotic/chlorotic lesion zones
            leaf_uint8 = leaf_raw_bool.astype(np.uint8) * 255
            contours, _ = cv2.findContours(leaf_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                filled_leaf = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(filled_leaf, [largest_c], -1, 255, thickness=cv2.FILLED)
                leaf_bool = filled_leaf > 0
            else:
                leaf_bool = leaf_raw_bool
        else:
            leaf_bool = np.zeros((h, w), dtype=bool)

        # If no leaf detected, return all-background mask
        if not np.any(leaf_bool):
            return np.zeros((h, w), dtype=np.uint8)

        # ---------------------------------------------------------------------
        # Step 2: Foliar Lesion Segmentation within the Leaf Lamina
        # ---------------------------------------------------------------------
        # CIELAB color space
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        # HSV color space
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Condition A: Necrotic dark-brown / black lesion centers (Sigatoka / Cordana)
        # Low luminance or elevated reddish a* with moderate luminance
        necrotic_condition = (
            leaf_bool &
            (
                ((a_channel >= 125) & (l_channel < 135)) |
                ((h_channel <= 25) & (v_channel < 130) & (s_channel > 50)) |
                ((l_channel < 60) & (v_channel < 75))
            )
        )

        # Condition B: Chlorotic yellow halos / marginal yellowing
        # Yellow hue range (H in [15, 32]), distinct from healthy green (H >= 35)
        chlorotic_condition = (
            leaf_bool &
            (
                (h_channel >= 15) & (h_channel <= 32) & (s_channel > 60) & (a_channel >= 114)
            )
        )

        # Combined lesion candidates
        lesion_raw = (necrotic_condition | chlorotic_condition).astype(np.uint8) * 255

        # Filter noise smaller than min_lesion_pixels
        num_les_labels, les_labels, les_stats, _ = cv2.connectedComponentsWithStats(lesion_raw, connectivity=8)
        lesion_clean = np.zeros_like(lesion_raw)
        for lbl in range(1, num_les_labels):
            if les_stats[lbl, cv2.CC_STAT_AREA] >= self.min_lesion_pixels:
                lesion_clean[les_labels == lbl] = 255

        lesion_bool = (lesion_clean > 0) & leaf_bool

        # ---------------------------------------------------------------------
        # Step 3: Construct 3-Class Output Mask
        # ---------------------------------------------------------------------
        mask_3class = np.zeros((h, w), dtype=np.uint8)
        mask_3class[leaf_bool] = 1   # Healthy leaf tissue
        mask_3class[lesion_bool] = 2 # Lesion tissue

        return mask_3class

    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Runs classical segmentation on a batch of RGB numpy arrays.
        """
        return [self.predict(img) for img in images]

    def predict_from_path(self, image_path: str) -> np.ndarray:
        """
        Loads an image from path and predicts the 3-class mask.
        """
        with Image.open(image_path) as img:
            rgb_np = np.array(img.convert("RGB"), dtype=np.uint8)
        return self.predict(rgb_np)
