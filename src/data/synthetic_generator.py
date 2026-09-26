"""
Longitudinal Synthetic Banana Leaf & Lesion Generator.

Generates:
1. 120 Cavendish banana plants distributed across 4 management zones (ZONE_A to ZONE_D).
2. 6 weekly inspection timepoints (t = 0, 7, 14, 21, 28, 35 days) = 720 multimodal records.
3. Photorealistic synthetic RGB leaf imagery (256x256) paired with pixel-exact ground-truth
   segmentation masks (0=background, 1=healthy leaf, 2=lesion tissue).
4. Disease-specific morphology:
   - Stunted aspect ratio (1.6 - 2.4) and vein-streaking for BBTV.
   - Slender aspect ratio (3.0 - 3.6) and elongated streaks for Sigatoka.
   - Large oval marginal lesions and chlorotic halos for Cordana.
5. Realistic longitudinal progression trajectories (Worsening, Stable, Improving).
"""

import os
import random
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import cv2
import pandas as pd
from PIL import Image

from src.schemas.contracts import DiseaseType, SeverityCategory, ProgressionTrend

# Constants & Paths
SYNTHETIC_DIR = r"d:\assignments\banana\banana_severity_c3\data\synthetic"
IMAGES_DIR = os.path.join(SYNTHETIC_DIR, "images")
MASKS_DIR = os.path.join(SYNTHETIC_DIR, "masks")
METADATA_PATH = os.path.join(SYNTHETIC_DIR, "metadata.csv")

IMG_SIZE = (256, 256)
RANDOM_SEED = 42

ZONES = ["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D"]
NUM_PLANTS = 120
TIMEPOINTS = [0, 7, 14, 21, 28, 35]  # Days


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def create_leaf_canvas(aspect_ratio: float, img_size: Tuple[int, int] = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renders a realistic green curved banana leaf lamina on a dark background.

    Args:
        aspect_ratio: Length to width ratio (e.g., 3.5 for healthy, 1.8 for stunted BBTV).
        img_size: (H, W) canvas size.

    Returns:
        canvas_rgb: (H, W, 3) uint8 image.
        leaf_mask: (H, W) bool array where leaf is present.
    """
    h, w = img_size
    canvas_rgb = np.full((h, w, 3), 18, dtype=np.uint8)  # Dark neutral background

    # Compute ellipse axes based on aspect ratio
    center = (w // 2, h // 2)
    major_axis = int(min(h, w) * 0.44)
    minor_axis = int(major_axis / max(1.0, aspect_ratio))
    minor_axis = max(18, minor_axis)

    angle = random.uniform(-15.0, 15.0)

    # Base green coloration (Cavendish lamina green)
    base_color = (
        random.randint(40, 65),   # Red
        random.randint(120, 165), # Green
        random.randint(25, 45)    # Blue
    )

    # Draw leaf lamina ellipse
    leaf_canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(leaf_canvas, center, (major_axis, minor_axis), angle, 0, 360, 255, -1)

    # Draw midrib (central vein) down the major axis
    midrib_mask = np.zeros((h, w), dtype=np.uint8)
    rad = np.radians(angle)
    dx = int(major_axis * np.cos(rad))
    dy = int(major_axis * np.sin(rad))
    pt1 = (center[0] - dx, center[1] - dy)
    pt2 = (center[0] + dx, center[1] + dy)
    cv2.line(midrib_mask, pt1, pt2, 255, thickness=2)

    leaf_bool = leaf_canvas > 0

    # Fill base leaf texture
    canvas_rgb[leaf_bool] = base_color

    # Add subtle lighting gradient across the lamina
    gradient = np.linspace(0.88, 1.12, w, dtype=np.float32)[np.newaxis, :, np.newaxis]
    canvas_rgb = np.clip(canvas_rgb.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    # Midrib lighter pale-green color
    midrib_bool = (midrib_mask > 0) & leaf_bool
    canvas_rgb[midrib_bool] = (
        min(255, base_color[0] + 35),
        min(255, base_color[1] + 40),
        min(255, base_color[2] + 25)
    )

    return canvas_rgb, leaf_bool


def render_lesions(
    canvas_rgb: np.ndarray,
    leaf_bool: np.ndarray,
    disease: DiseaseType,
    severity_target: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Renders disease-specific lesions inside the leaf lamina.

    Args:
        canvas_rgb: Base leaf image.
        leaf_bool: Leaf mask.
        disease: DiseaseType.
        severity_target: Desired target scope percentage in [0, 100].

    Returns:
        canvas_rgb: Image with lesions painted.
        lesion_bool: Boolean mask of lesion pixels.
        metrics: Dictionary of actual computed lesion metrics.
    """
    h, w, _ = canvas_rgb.shape
    lesion_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_pixel_count = np.count_nonzero(leaf_bool)

    if leaf_pixel_count == 0 or disease == DiseaseType.HEALTHY or severity_target <= 0.0:
        return canvas_rgb, np.zeros((h, w), dtype=bool), {
            "num_lesions": 0, "scope_pct": 0.0, "coalescence_ratio": 0.0
        }

    target_lesion_pixels = int(leaf_pixel_count * (severity_target / 100.0))
    target_lesion_pixels = max(10, min(int(leaf_pixel_count * 0.85), target_lesion_pixels))

    current_lesion_pixels = 0
    num_lesions = 0
    max_single_lesion = 0

    leaf_indices = np.argwhere(leaf_bool)
    if len(leaf_indices) == 0:
        return canvas_rgb, np.zeros((h, w), dtype=bool), {"num_lesions": 0, "scope_pct": 0.0, "coalescence_ratio": 0.0}

    # Disease specific lesion painting
    if disease == DiseaseType.BLACK_SIGATOKA:
        # Elongated dark brown / black streaks oriented along vein angles
        while current_lesion_pixels < target_lesion_pixels and num_lesions < 80:
            idx = random.randint(0, len(leaf_indices) - 1)
            cy, cx = leaf_indices[idx]
            streak_len = random.randint(6, 24)
            streak_w = random.randint(2, 4)
            angle = random.uniform(30.0, 60.0)

            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(temp_mask, (cx, cy), (streak_len, streak_w), angle, 0, 360, 255, -1)
            temp_mask = temp_mask & (leaf_bool.astype(np.uint8) * 255)

            pixels_added = np.count_nonzero(temp_mask & ~lesion_mask)
            if pixels_added > 0:
                lesion_mask = lesion_mask | temp_mask
                current_lesion_pixels = np.count_nonzero(lesion_mask)
                num_lesions += 1
                max_single_lesion = max(max_single_lesion, np.count_nonzero(temp_mask))

                # Color: Dark necrotic brown/black with subtle reddish tinge
                color = (random.randint(45, 75), random.randint(25, 45), random.randint(15, 30))
                canvas_rgb[temp_mask > 0] = color

    elif disease == DiseaseType.YELLOW_SIGATOKA:
        # Pale yellow-brown streaks with chlorotic borders
        while current_lesion_pixels < target_lesion_pixels and num_lesions < 75:
            idx = random.randint(0, len(leaf_indices) - 1)
            cy, cx = leaf_indices[idx]
            streak_len = random.randint(6, 20)
            streak_w = random.randint(2, 5)
            angle = random.uniform(25.0, 55.0)

            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(temp_mask, (cx, cy), (streak_len, streak_w), angle, 0, 360, 255, -1)
            temp_mask = temp_mask & (leaf_bool.astype(np.uint8) * 255)

            pixels_added = np.count_nonzero(temp_mask & ~lesion_mask)
            if pixels_added > 0:
                lesion_mask = lesion_mask | temp_mask
                current_lesion_pixels = np.count_nonzero(lesion_mask)
                num_lesions += 1
                max_single_lesion = max(max_single_lesion, np.count_nonzero(temp_mask))

                # Color: Pale yellow-brown / rust
                color = (random.randint(160, 195), random.randint(130, 160), random.randint(35, 65))
                canvas_rgb[temp_mask > 0] = color

    elif disease == DiseaseType.CORDANA:
        # Fewer large oval-to-diamond marginal lesions with bright yellow halos
        num_targets = max(1, int(severity_target / 12.0))
        for _ in range(num_targets):
            if current_lesion_pixels >= target_lesion_pixels:
                break
            idx = random.randint(0, len(leaf_indices) - 1)
            cy, cx = leaf_indices[idx]
            radius = random.randint(12, 28)

            # Outer yellow halo
            halo_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(halo_mask, (cx, cy), int(radius * 1.35), 255, -1)
            halo_mask = halo_mask & (leaf_bool.astype(np.uint8) * 255)
            canvas_rgb[halo_mask > 0] = (random.randint(200, 230), random.randint(190, 220), 40)

            # Inner necrotic core
            core_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(core_mask, (cx, cy), radius, 255, -1)
            core_mask = core_mask & (leaf_bool.astype(np.uint8) * 255)
            canvas_rgb[core_mask > 0] = (random.randint(70, 95), random.randint(40, 60), 25)

            combined = halo_mask | core_mask
            lesion_mask = lesion_mask | combined
            current_lesion_pixels = np.count_nonzero(lesion_mask)
            num_lesions += 1
            max_single_lesion = max(max_single_lesion, np.count_nonzero(combined))

    elif disease == DiseaseType.BBTV:
        # Marginal chlorosis + dark-green Morse-code vein streaks
        # 1. Marginal chlorosis
        margin_dist = cv2.distanceTransform(leaf_bool.astype(np.uint8), cv2.DIST_L2, 5)
        margin_mask = (margin_dist > 0) & (margin_dist < 6)
        canvas_rgb[margin_mask] = (random.randint(185, 215), random.randint(185, 210), 30)
        lesion_mask = lesion_mask | (margin_mask.astype(np.uint8) * 255)

        # 2. Dark green vein streaks
        while current_lesion_pixels < target_lesion_pixels and num_lesions < 60:
            idx = random.randint(0, len(leaf_indices) - 1)
            cy, cx = leaf_indices[idx]
            slen = random.randint(4, 12)
            cv2.line(lesion_mask, (cx, cy), (cx + slen, cy), 255, thickness=2)
            lesion_mask = lesion_mask & (leaf_bool.astype(np.uint8) * 255)
            current_lesion_pixels = np.count_nonzero(lesion_mask)
            num_lesions += 1

            # Dark-green / purplish vein streaking
            streak_bool = lesion_mask > 0
            canvas_rgb[streak_bool] = (random.randint(20, 35), random.randint(70, 95), random.randint(25, 45))

    lesion_bool = (lesion_mask > 0) & leaf_bool
    actual_scope = (np.count_nonzero(lesion_bool) / leaf_pixel_count) * 100.0
    coalescence_ratio = (max_single_lesion / leaf_pixel_count) * 100.0

    # Add subtle Gaussian sensor noise
    noise = np.random.normal(0, 3.5, canvas_rgb.shape).astype(np.float32)
    canvas_noisy = np.clip(canvas_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return canvas_noisy, lesion_bool, {
        "num_lesions": num_lesions,
        "scope_pct": round(actual_scope, 2),
        "coalescence_ratio": round(coalescence_ratio, 2)
    }


def generate_synthetic_cohort(
    num_plants: int = NUM_PLANTS,
    output_dir: str = SYNTHETIC_DIR,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generates 120 plants tracked across 6 weekly timepoints (720 records),
    saving RGB images, ground-truth masks, and metadata.csv.
    """
    set_seed(seed)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

    records = []
    base_date = datetime(2026, 8, 1)

    disease_distribution = [
        DiseaseType.BLACK_SIGATOKA,
        DiseaseType.YELLOW_SIGATOKA,
        DiseaseType.CORDANA,
        DiseaseType.BBTV,
        DiseaseType.HEALTHY
    ]

    print(f"Generating synthetic cohort: {num_plants} plants across {len(TIMEPOINTS)} weeks...")

    for p_idx in range(1, num_plants + 1):
        plant_id = f"PLANT_{p_idx:03d}"
        zone_id = ZONES[(p_idx - 1) % len(ZONES)]
        disease = disease_distribution[(p_idx - 1) % len(disease_distribution)]

        # Determine plant trajectory type
        if disease == DiseaseType.HEALTHY:
            traj_type = ProgressionTrend.STABLE
            base_severity = 0.0
        else:
            traj_choice = random.random()
            if traj_choice < 0.50:
                traj_type = ProgressionTrend.WORSENING
            elif traj_choice < 0.80:
                traj_type = ProgressionTrend.STABLE
            else:
                traj_type = ProgressionTrend.IMPROVING

            base_severity = random.uniform(8.0, 32.0)

        # Baseline plant leaf aspect ratio (stunted for BBTV)
        if disease == DiseaseType.BBTV:
            plant_aspect_ratio = random.uniform(1.65, 2.35)
        else:
            plant_aspect_ratio = random.uniform(3.10, 3.55)

        # Environmental urgency score for the plant's zone
        zone_urgency_base = {
            "ZONE_A": 72.0,  # High risk zone
            "ZONE_B": 48.0,  # Moderate risk zone
            "ZONE_C": 82.0,  # High / Critical risk zone
            "ZONE_D": 30.0   # Low / Moderate risk zone
        }[zone_id]

        current_severity = base_severity

        for t_idx, days in enumerate(TIMEPOINTS):
            record_id = f"{plant_id}_T{t_idx}_D{days:02d}"
            obs_date = (base_date + timedelta(days=days)).strftime("%Y-%m-%d")

            # Evolve severity based on trajectory
            if disease == DiseaseType.HEALTHY:
                current_severity = 0.0
            else:
                if traj_type == ProgressionTrend.WORSENING:
                    step = random.uniform(4.5, 9.5)
                    current_severity = min(92.0, current_severity + step)
                elif traj_type == ProgressionTrend.STABLE:
                    jitter = random.uniform(-1.8, 1.8)
                    current_severity = max(3.0, min(80.0, current_severity + jitter))
                elif traj_type == ProgressionTrend.IMPROVING:
                    step = random.uniform(2.5, 6.0)
                    current_severity = max(2.0, current_severity - step)

            # Environmental urgency with weekly variation
            urgency_score = round(max(0.0, min(100.0, zone_urgency_base + random.uniform(-5.0, 5.0))), 1)

            # 1. Render leaf canvas
            canvas, leaf_bool = create_leaf_canvas(plant_aspect_ratio, IMG_SIZE)

            # 2. Render lesions
            canvas_noisy, lesion_bool, metrics = render_lesions(
                canvas, leaf_bool, disease, current_severity
            )

            # 3. Construct 3-class ground truth mask
            gt_mask = np.zeros(IMG_SIZE, dtype=np.uint8)
            gt_mask[leaf_bool] = 1
            gt_mask[lesion_bool] = 2

            # 4. Save image and mask
            img_filename = f"{record_id}.jpg"
            mask_filename = f"{record_id}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            mask_path = os.path.join(MASKS_DIR, mask_filename)

            Image.fromarray(canvas_noisy).save(img_path, quality=95)
            Image.fromarray(gt_mask).save(mask_path)

            # Calculate Stover-Gauhl grade for Sigatoka
            scope = metrics["scope_pct"]
            if disease in [DiseaseType.BLACK_SIGATOKA, DiseaseType.YELLOW_SIGATOKA]:
                if scope == 0.0: stover_b = 0
                elif scope <= 5.0: stover_b = 1
                elif scope <= 13.0: stover_b = 2
                elif scope <= 23.0: stover_b = 3
                elif scope <= 40.0: stover_b = 4
                elif scope <= 65.0: stover_b = 5
                else: stover_b = 6
            else:
                stover_b = None

            records.append({
                "record_id": record_id,
                "plant_id": plant_id,
                "zone_id": zone_id,
                "leaf_id": f"LEAF_{p_idx:03d}",
                "timepoint_idx": t_idx,
                "day_index": days,
                "observation_date": obs_date,
                "disease_label": disease.value,
                "image_path": os.path.relpath(img_path, SYNTHETIC_DIR).replace("\\", "/"),
                "mask_path": os.path.relpath(mask_path, SYNTHETIC_DIR).replace("\\", "/"),
                "ground_truth_scope_pct": scope,
                "ground_truth_aspect_ratio": round(plant_aspect_ratio, 2),
                "num_lesions": metrics["num_lesions"],
                "coalescence_ratio_pct": metrics["coalescence_ratio"],
                "trajectory_type": traj_type.value,
                "urgency_score": urgency_score,
                "stover_grade": stover_b
            })

    df = pd.DataFrame(records)
    df.to_csv(METADATA_PATH, index=False)
    print(f"Synthetic cohort generation complete! {len(df)} records saved to {METADATA_PATH}")
    return df


if __name__ == "__main__":
    generate_synthetic_cohort()
