import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


INPUT_CSV = Path(
    "results/confidence_calibration/validation_distances.csv"
)

OUTPUT_DIR = Path("results/confidence_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "black_sigatoka",
    "healthy",
    "yellow_sigatoka",
]

# Use float64 for numerical stability.
data = pd.read_csv(INPUT_CSV)

distance_columns = [
    f"distance_{class_name}"
    for class_name in CLASS_NAMES
]

distances = torch.tensor(
    data[distance_columns].to_numpy(),
    dtype=torch.float64,
)

labels = torch.tensor(
    data["true_label"].to_numpy(),
    dtype=torch.long,
)

# Smaller distances should produce larger scores.
logits = -distances

# --------------------------------------------------
# Fit temperature on validation data
# --------------------------------------------------

log_temperature = torch.nn.Parameter(
    torch.zeros((), dtype=torch.float64)
)

optimizer = torch.optim.LBFGS(
    [log_temperature],
    lr=0.1,
    max_iter=200,
    line_search_fn="strong_wolfe",
)

def closure():
    optimizer.zero_grad()

    temperature = torch.exp(log_temperature)

    loss = F.cross_entropy(
        logits / temperature,
        labels,
    )

    loss.backward()
    return loss

optimizer.step(closure)

temperature = torch.exp(
    log_temperature.detach()
).item()

# Avoid a numerically unusable temperature.
if not np.isfinite(temperature) or temperature <= 0:
    raise RuntimeError(
        f"Invalid fitted temperature: {temperature}"
    )

# --------------------------------------------------
# Calculate scores before and after calibration
# --------------------------------------------------

with torch.no_grad():

    raw_scores = torch.softmax(logits, dim=1)

    calibrated_scores = torch.softmax(
        logits / temperature,
        dim=1,
    )

    predictions = logits.argmax(dim=1)

    raw_confidence = raw_scores.max(dim=1).values
    calibrated_confidence = (
        calibrated_scores.max(dim=1).values
    )

    correct = predictions.eq(labels)

# --------------------------------------------------
# Expected Calibration Error (ECE)
# --------------------------------------------------

def expected_calibration_error(
    confidence,
    correctness,
    bins=5,
):
    confidence = confidence.numpy()
    correctness = correctness.numpy().astype(float)

    ece = 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)

    for i in range(bins):

        # Left-inclusive; right-exclusive, except last bin.
        if i == bins - 1:
            mask = (
                (confidence >= edges[i])
                & (confidence <= edges[i + 1])
            )
        else:
            mask = (
                (confidence >= edges[i])
                & (confidence < edges[i + 1])
            )

        if mask.any():

            bin_accuracy = correctness[mask].mean()
            bin_confidence = confidence[mask].mean()

            ece += (
                mask.mean()
                * abs(bin_accuracy - bin_confidence)
            )

    return float(ece)


def brier_score(probabilities, labels, num_classes):
    targets = F.one_hot(
        labels,
        num_classes=num_classes,
    ).to(probabilities.dtype)

    return float(
        ((probabilities - targets) ** 2)
        .sum(dim=1)
        .mean()
    )


raw_ece = expected_calibration_error(
    raw_confidence,
    correct,
)

calibrated_ece = expected_calibration_error(
    calibrated_confidence,
    correct,
)

raw_brier = brier_score(
    raw_scores,
    labels,
    len(CLASS_NAMES),
)

calibrated_brier = brier_score(
    calibrated_scores,
    labels,
    len(CLASS_NAMES),
)

# --------------------------------------------------
# Save validation scores
# --------------------------------------------------

output = data.copy()

for i, class_name in enumerate(CLASS_NAMES):
    output[f"raw_score_{class_name}"] = (
        raw_scores[:, i].numpy()
    )

    output[f"calibrated_score_{class_name}"] = (
        calibrated_scores[:, i].numpy()
    )

output["raw_confidence"] = raw_confidence.numpy()
output["calibrated_confidence"] = (
    calibrated_confidence.numpy()
)
output["correct_prediction"] = correct.numpy()

output.to_csv(
    OUTPUT_DIR / "validation_confidence_scores.csv",
    index=False,
)

summary = {
    "validation_images": len(data),
    "validation_accuracy": float(correct.double().mean()),
    "temperature": temperature,
    "raw_ece_5_bins": raw_ece,
    "calibrated_ece_5_bins": calibrated_ece,
    "raw_multiclass_brier_score": raw_brier,
    "calibrated_multiclass_brier_score": calibrated_brier,
    "warning": (
        "Calibration metrics are measured on the same validation "
        "set used to fit temperature. They are exploratory and may "
        "be optimistic, especially with only 60 images."
    ),
}

with open(
    OUTPUT_DIR / "calibration_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)

print("\nConfidence Calibration Results")
print("--------------------------------")
print(f"Validation images: {len(data)}")
print(f"Validation accuracy: {correct.double().mean().item():.4f}")
print(f"Fitted temperature: {temperature:.6f}")

print(f"\nRaw ECE (5 bins): {raw_ece:.4f}")
print(f"Calibrated ECE (5 bins): {calibrated_ece:.4f}")

print(f"\nRaw Brier score: {raw_brier:.4f}")
print(f"Calibrated Brier score: {calibrated_brier:.4f}")

print("\nSaved:")
print(OUTPUT_DIR / "validation_confidence_scores.csv")
print(OUTPUT_DIR / "calibration_summary.json")