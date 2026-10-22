import json
from pathlib import Path

import pandas as pd


SUMMARY_PATH = Path(
    "results/confidence_calibration/calibration_summary.json"
)

OUTPUT_PATH = Path(
    "models/resnet18_confidence_calibration.json"
)

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(
        f"Calibration summary not found: {SUMMARY_PATH}"
    )

with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
    summary = json.load(file)

temperature = float(summary["temperature"])

if temperature <= 0:
    raise ValueError("Temperature must be positive.")

artifact = {
    "method": "temperature_scaling",
    "temperature": temperature,
    "num_calibration_images": summary["validation_images"],
    "calibration_accuracy": summary["validation_accuracy"],
    "raw_ece_5_bins": summary["raw_ece_5_bins"],
    "calibrated_ece_5_bins": summary["calibrated_ece_5_bins"],
    "raw_multiclass_brier_score": summary["raw_multiclass_brier_score"],
    "calibrated_multiclass_brier_score": summary[
        "calibrated_multiclass_brier_score"
    ],
    "score_interpretation": (
        "Temperature-scaled model score; not a guaranteed "
        "probability that the prediction is correct."
    ),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(artifact, file, indent=4)

print("Calibration artifact saved:", OUTPUT_PATH)
print("Temperature:", temperature)
print("Calibration images:", summary["validation_images"])