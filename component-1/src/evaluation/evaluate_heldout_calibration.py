import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_CSV = Path(
    "results/confidence_calibration/validation_distances.csv"
)

OUTPUT_DIR = Path(
    "results/confidence_calibration/heldout"
)

CLASS_NAMES = [
    "black_sigatoka",
    "healthy",
    "yellow_sigatoka",
]

CALIBRATION_FRACTION = 0.5
SEED = 42
NUM_BINS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load validation distances
# --------------------------------------------------

df = pd.read_csv(INPUT_CSV)

distance_columns = [
    f"distance_{name}"
    for name in CLASS_NAMES
]

distances = torch.tensor(
    df[distance_columns].to_numpy(),
    dtype=torch.float64,
)

labels = torch.tensor(
    df["true_label"].to_numpy(),
    dtype=torch.long,
)

# Smaller distance means a closer prototype.
logits = -distances

indices = np.arange(len(df))

calibration_indices, evaluation_indices = train_test_split(
    indices,
    test_size=1 - CALIBRATION_FRACTION,
    random_state=SEED,
    stratify=labels.numpy(),
)

calibration_indices = torch.tensor(
    calibration_indices,
    dtype=torch.long,
)

evaluation_indices = torch.tensor(
    evaluation_indices,
    dtype=torch.long,
)

calibration_logits = logits[calibration_indices]
calibration_labels = labels[calibration_indices]

evaluation_logits = logits[evaluation_indices]
evaluation_labels = labels[evaluation_indices]


# --------------------------------------------------
# Fit temperature on calibration subset only
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
        calibration_logits / temperature,
        calibration_labels,
    )

    loss.backward()

    return loss


optimizer.step(closure)

temperature = torch.exp(
    log_temperature.detach()
).item()

if not np.isfinite(temperature) or temperature <= 0:
    raise RuntimeError(
        f"Invalid fitted temperature: {temperature}"
    )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def expected_calibration_error(
    confidence,
    correctness,
    bins=5,
):
    confidence = confidence.detach().cpu().numpy()
    correctness = (
        correctness.detach().cpu().numpy().astype(float)
    )

    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)

    for i in range(bins):

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

            accuracy = correctness[mask].mean()
            average_confidence = confidence[mask].mean()

            ece += (
                mask.mean()
                * abs(accuracy - average_confidence)
            )

    return float(ece)


def multiclass_brier_score(probabilities, labels):
    targets = F.one_hot(
        labels,
        num_classes=len(CLASS_NAMES),
    ).to(probabilities.dtype)

    return float(
        ((probabilities - targets) ** 2)
        .sum(dim=1)
        .mean()
    )


def calculate_metrics(logits, labels, temperature):
    with torch.no_grad():

        raw_probabilities = torch.softmax(
            logits,
            dim=1,
        )

        calibrated_probabilities = torch.softmax(
            logits / temperature,
            dim=1,
        )

        predictions = logits.argmax(dim=1)
        correct = predictions.eq(labels)

        raw_confidence = raw_probabilities.max(dim=1).values

        calibrated_confidence = (
            calibrated_probabilities.max(dim=1).values
        )

        return {
            "count": len(labels),
            "accuracy": float(correct.double().mean()),
            "raw_ece": expected_calibration_error(
                raw_confidence,
                correct,
                NUM_BINS,
            ),
            "calibrated_ece": expected_calibration_error(
                calibrated_confidence,
                correct,
                NUM_BINS,
            ),
            "raw_brier": multiclass_brier_score(
                raw_probabilities,
                labels,
            ),
            "calibrated_brier": multiclass_brier_score(
                calibrated_probabilities,
                labels,
            ),
            "mean_raw_confidence": float(
                raw_confidence.mean()
            ),
            "mean_calibrated_confidence": float(
                calibrated_confidence.mean()
            ),
        }, calibrated_probabilities, correct


# --------------------------------------------------
# Evaluate both subsets
# --------------------------------------------------

calibration_metrics, _, _ = calculate_metrics(
    calibration_logits,
    calibration_labels,
    temperature,
)

evaluation_metrics, evaluation_probabilities, evaluation_correct = (
    calculate_metrics(
        evaluation_logits,
        evaluation_labels,
        temperature,
    )
)


# --------------------------------------------------
# Save evaluation predictions
# --------------------------------------------------

evaluation_df = df.iloc[
    evaluation_indices.numpy()
].copy()

evaluation_df["correct_prediction"] = (
    evaluation_correct.cpu().numpy()
)

evaluation_df["calibrated_confidence"] = (
    evaluation_probabilities.max(dim=1)
    .values.cpu().numpy()
)

for i, class_name in enumerate(CLASS_NAMES):
    evaluation_df[f"calibrated_score_{class_name}"] = (
        evaluation_probabilities[:, i].cpu().numpy()
    )

evaluation_df.to_csv(
    OUTPUT_DIR / "heldout_predictions.csv",
    index=False,
)


# --------------------------------------------------
# Save summary
# --------------------------------------------------

summary = {
    "seed": SEED,
    "calibration_images": len(calibration_indices),
    "evaluation_images": len(evaluation_indices),
    "temperature": temperature,
    "calibration_subset_metrics": calibration_metrics,
    "heldout_evaluation_metrics": evaluation_metrics,
    "warning": (
        "The held-out subset is small. Its metrics are exploratory "
        "and uncertain. The temperature was fitted using the "
        "calibration subset only. Neither subset is the original "
        "33-image test set."
    ),
}

with open(
    OUTPUT_DIR / "heldout_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nHELD-OUT CALIBRATION EVALUATION")
print("--------------------------------")
print(f"Calibration images: {len(calibration_indices)}")
print(f"Evaluation images: {len(evaluation_indices)}")
print(f"Fitted temperature: {temperature:.6f}")

print("\nCalibration subset:")
for key, value in calibration_metrics.items():
    print(f"{key}: {value}")

print("\nHeld-out evaluation subset:")
for key, value in evaluation_metrics.items():
    print(f"{key}: {value}")

print("\nSaved:")
print(OUTPUT_DIR / "heldout_predictions.csv")
print(OUTPUT_DIR / "heldout_summary.json")