import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.transforms import validation_transforms
from src.models.feature_extractor import create_feature_extractor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROTOTYPE_PATH = Path(
    "models/resnet18_prototypes_20shot_seed42.pth"
)

VALIDATION_DIR = Path("data/processed/validation")

OUTPUT_DIR = Path("results/confidence_calibration")

BATCH_SIZE = 16
SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Setup
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not PROTOTYPE_PATH.exists():
    raise FileNotFoundError(
        f"Prototype artifact not found: {PROTOTYPE_PATH}"
    )

if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found: {VALIDATION_DIR}"
    )


# --------------------------------------------------
# Load saved prototypes
# --------------------------------------------------

artifact = torch.load(
    PROTOTYPE_PATH,
    map_location="cpu",
    weights_only=False,
)

prototypes = artifact["prototypes"].to(device)
class_names = artifact["class_names"]

print("Classes:", class_names)
print("Device:", device)


# --------------------------------------------------
# Load validation dataset
# --------------------------------------------------

dataset = ImageFolder(
    root=str(VALIDATION_DIR),
    transform=validation_transforms,
)

if dataset.classes != class_names:
    raise ValueError(
        f"Class mismatch. Dataset: {dataset.classes}; "
        f"Prototypes: {class_names}"
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# Load frozen feature extractor
# --------------------------------------------------

model = create_feature_extractor().to(device)
model.eval()


# --------------------------------------------------
# Calculate validation distances
# --------------------------------------------------

records = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        features = model(images)
        features = features.flatten(start_dim=1)

        embeddings = F.normalize(
            features,
            p=2,
            dim=1,
        )

        distances = torch.cdist(
            embeddings,
            prototypes,
        )

        predictions = distances.argmin(dim=1)

        for i in range(len(labels)):

            row = {
                "image_path": dataset.samples[
                    len(records)
                ][0],
                "true_class": class_names[
                    labels[i].item()
                ],
                "predicted_class": class_names[
                    predictions[i].item()
                ],
                "true_label": labels[i].item(),
                "predicted_label": predictions[i].item(),
            }

            for class_index, class_name in enumerate(
                class_names
            ):
                row[f"distance_{class_name}"] = float(
                    distances[i, class_index].item()
                )

            records.append(row)


# --------------------------------------------------
# Save validation distances
# --------------------------------------------------

df = pd.DataFrame(records)

csv_path = OUTPUT_DIR / "validation_distances.csv"
df.to_csv(csv_path, index=False)

correct = (
    df["true_label"] == df["predicted_label"]
).sum()

accuracy = correct / len(df)

summary = {
    "validation_images": len(df),
    "correct_predictions": int(correct),
    "incorrect_predictions": int(len(df) - correct),
    "accuracy": float(accuracy),
    "classes": class_names,
    "prototype_path": str(PROTOTYPE_PATH),
}

summary_path = OUTPUT_DIR / "validation_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nValidation results")
print("------------------")
print(f"Images: {len(df)}")
print(f"Correct: {correct}")
print(f"Incorrect: {len(df) - correct}")
print(f"Accuracy: {accuracy:.4f}")

print("\nSaved:")
print(csv_path)
print(summary_path)