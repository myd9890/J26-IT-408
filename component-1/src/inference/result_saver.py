import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("results/inference")


def save_inference_result(
    result,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Save inference metadata and optional Grad-CAM explanation.

    Supports:
        1. Successful disease predictions.
        2. Low-confidence predictions requiring review.
        3. Images rejected or marked uncertain by the CLIP gatekeeper.

    Returns:
        Path to the directory containing the saved outputs.
    """

    output_dir = Path(output_dir)
    image_path = Path(result["image_path"])

    # Create a safe directory name from the image filename.
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        image_path.stem,
    )

    result_dir = output_dir / safe_name
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Save prediction metadata.
    # --------------------------------------------------

    prediction = {
        "image_path": str(image_path),
        "predicted_class": result.get("predicted_class"),
        "status": result.get("status"),
    }

    # Include an explanatory message for rejected/uncertain images.
    if "message" in result:
        prediction["message"] = result["message"]

    # Include gatekeeper information whenever available.
    if "gatekeeper" in result:
        prediction["gatekeeper"] = result["gatekeeper"]

    # Include classifier fields only when classification occurred.
    if result.get("predicted_class") is not None:

        prediction["model_confidence_score"] = float(
            result["model_confidence_score"]
        )

        prediction["model_confidence_percentage"] = float(
            result["model_confidence_percentage"]
        )

        prediction["class_scores"] = {
            name: float(score)
            for name, score in result["class_scores"].items()
        }

        prediction["distances"] = {
            name: float(distance)
            for name, distance in result["distances"].items()
        }

        prediction["nearest_distance_margin"] = float(
            result["nearest_distance_margin"]
        )

    if "explanation_target" in result:
        prediction["explanation_target"] = (
            result["explanation_target"]
        )

    prediction_path = result_dir / "prediction.json"

    with prediction_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            prediction,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # Save explanation images, if available.
    # --------------------------------------------------

    if "gradcam" in result and "display_image" in result:

        original = np.asarray(result["display_image"])
        heatmap = np.asarray(result["gradcam"])

        if heatmap.shape != original.shape[:2]:
            raise ValueError(
                "Grad-CAM dimensions do not match the display image."
            )

        if not np.isfinite(heatmap).all():
            raise ValueError(
                "Grad-CAM contains invalid numerical values."
            )

        # Save the original display image.
        original_path = result_dir / "original_image.png"

        plt.imsave(
            original_path,
            original,
        )

        # Save the standalone heatmap.
        heatmap_path = result_dir / "gradcam_heatmap.png"

        plt.imsave(
            heatmap_path,
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1,
        )

        # Save the overlay.
        overlay_path = result_dir / "gradcam_overlay.png"

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(original)

        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        ax.axis("off")
        fig.tight_layout(pad=0)

        fig.savefig(
            overlay_path,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close(fig)

    print("\nInference results saved to:")
    print(result_dir)

    return result_dir

