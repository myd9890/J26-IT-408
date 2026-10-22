import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

resnet_path = RESULTS_DIR / "resnet18_few_shot_runs.csv"
dino_path = RESULTS_DIR / "dinov2_few_shot_runs.csv"

resnet = pd.read_csv(resnet_path)
dino = pd.read_csv(dino_path)

# Check that both experiments use the same support images.
merged = resnet.merge(
    dino,
    on=["shots", "seed"],
    suffixes=("_resnet", "_dino"),
    validate="one_to_one"
)

if len(merged) != 20:
    raise ValueError(
        f"Expected 20 matched runs, found {len(merged)}."
    )

mismatches = merged[
    merged["support_signature_resnet"]
    != merged["support_signature_dino"]
]

if not mismatches.empty:
    raise ValueError(
        "Support-set mismatch detected. "
        "Do not compare the results yet.\n"
        + mismatches[
            ["shots", "seed"]
        ].to_string(index=False)
    )

print("Support-set verification passed.")

# Calculate comparable population means and standard deviations.
resnet_summary = (
    resnet.groupby("shots")["accuracy"]
    .agg(["mean", "std"])
)

dino_summary = (
    dino.groupby("shots")["accuracy"]
    .agg(["mean", "std"])
)

comparison = pd.DataFrame({
    "shots": resnet_summary.index,
    "ResNet18_mean": resnet_summary["mean"].values * 100,
    "ResNet18_std": (
        resnet.groupby("shots")["accuracy"]
        .apply(lambda x: x.std(ddof=0))
        .values * 100
    ),
    "DINOv2_mean": dino_summary["mean"].values * 100,
    "DINOv2_std": (
        dino.groupby("shots")["accuracy"]
        .apply(lambda x: x.std(ddof=0))
        .values * 100
    )
})

print("\n========== FEW-SHOT COMPARISON ==========")
print(comparison.round(2).to_string(index=False))

comparison.to_csv(
    RESULTS_DIR / "few_shot_comparison.csv",
    index=False
)

print("\nComparison saved to results/few_shot_comparison.csv")