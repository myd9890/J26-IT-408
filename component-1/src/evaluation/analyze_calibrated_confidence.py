import pandas as pd
from pathlib import Path

path = Path(
    "results/confidence_calibration/validation_confidence_scores.csv"
)

df = pd.read_csv(path)

confidence = df["calibrated_confidence"]

print("\nCALIBRATED CONFIDENCE DISTRIBUTION")
print("---------------------------------")
print(f"Minimum: {confidence.min():.4f}")
print(f"25th percentile: {confidence.quantile(0.25):.4f}")
print(f"Median: {confidence.median():.4f}")
print(f"75th percentile: {confidence.quantile(0.75):.4f}")
print(f"Maximum: {confidence.max():.4f}")
print(f"Mean: {confidence.mean():.4f}")

print("\nCONFIDENCE BY PREDICTION CORRECTNESS")
print("------------------------------------")

for correct, group in df.groupby("correct_prediction"):
    label = "Correct predictions" if correct else "Incorrect predictions"

    print(f"\n{label}: {len(group)} images")
    print(f"Mean confidence: {group['calibrated_confidence'].mean():.4f}")
    print(f"Minimum confidence: {group['calibrated_confidence'].min():.4f}")
    print(f"Maximum confidence: {group['calibrated_confidence'].max():.4f}")

print("\nCALIBRATED CONFIDENCE BY TRUE CLASS")
print("-----------------------------------")

print(
    df.groupby("true_class")["calibrated_confidence"]
    .agg(["count", "mean", "min", "max"])
    .round(4)
)

print("\nCONFIDENT BUT INCORRECT PREDICTIONS")
print("-----------------------------------")

incorrect = df[df["correct_prediction"] == False].copy()

columns = [
    "image_path",
    "true_class",
    "predicted_class",
    "calibrated_confidence",
]

print(
    incorrect.sort_values(
        "calibrated_confidence",
        ascending=False,
    )[columns].to_string(index=False)
)