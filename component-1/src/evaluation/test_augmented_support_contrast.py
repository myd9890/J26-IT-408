import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.data.transforms import validation_transforms
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import create_prototypes


# ============================================================
# EXPERIMENT
# Compare original support images with contrast-augmented
# support images. Test images remain completely unchanged.
#
# This is a prototype-construction experiment, not model training:
# the feature extractor remains frozen.
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHOTS = 20
SEEDS = [42, 123, 456, 789, 1010]
BATCH_SIZE = 16

TRAIN_PATH = Path("data/processed/train")
TEST_PATH = Path("data/raw/test")
OUTPUT_DIR = Path("results/augmented_support_contrast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["black_sigatoka", "healthy", "yellow_sigatoka"]
CONTRAST_FACTOR = 1.30


def get_support_indices(dataset, shots, seed):
    """Match the support sampling logic used in evaluate_few_shot.py."""
    random.seed(seed)
    support_indices = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            idx for idx, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]
        support_indices.extend(random.sample(class_indices, shots))

    return support_indices


def extract_embeddings(model, dataset, indices, batch_size=16):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    embeddings, labels = [], []
    model.eval()

    with torch.no_grad():
        for images, batch_labels in loader:
            features = model(images.to(DEVICE)).flatten(start_dim=1)
            features = F.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu())
            labels.append(batch_labels)

    return torch.cat(embeddings), torch.cat(labels)


class ContrastAugmentedSupport(torch.utils.data.Dataset):
    """
    Deterministically applies the same 1.30 contrast enhancement to each
    selected support image. The selected image identities are unchanged.
    """

    def __init__(self, base_dataset, indices, contrast_factor):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.contrast_factor = contrast_factor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        original_index = self.indices[item]
        image_path, label = self.base_dataset.samples[original_index]

        image = Image.open(image_path).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(self.contrast_factor)
        tensor = validation_transforms(image)

        return tensor, label


def evaluate_seed(model, test_dataset, prototypes, seed, condition):
    rows = []
    model.eval()

    with torch.no_grad():
        for index, (image_tensor, actual_id) in enumerate(test_dataset):
            embedding = model(image_tensor.unsqueeze(0).to(DEVICE))
            embedding = F.normalize(embedding.flatten(start_dim=1), p=2, dim=1)
            distances = torch.cdist(embedding.cpu(), prototypes.cpu(), p=2)[0]
            predicted_id = int(distances.argmin().item())

            image_name = Path(test_dataset.samples[index][0]).name
            rows.append({
                "seed": seed,
                "support_condition": condition,
                "image": image_name,
                "actual_class": CLASS_NAMES[actual_id],
                "predicted_class": CLASS_NAMES[predicted_id],
                "correct": predicted_id == actual_id,
                "distance_black_sigatoka": float(distances[0]),
                "distance_healthy": float(distances[1]),
                "distance_yellow_sigatoka": float(distances[2]),
            })

    return rows


def summarize_records(records):
    per_seed = []

    for seed in SEEDS:
        for condition in ["original_support", "contrast_augmented_support"]:
            rows = [
                row for row in records
                if row["seed"] == seed
                and row["support_condition"] == condition
            ]

            summary = {
                "seed": seed,
                "support_condition": condition,
                "accuracy": float(np.mean([row["correct"] for row in rows])),
            }

            for class_name in CLASS_NAMES:
                class_rows = [
                    row for row in rows
                    if row["actual_class"] == class_name
                ]
                summary[f"recall_{class_name}"] = float(np.mean([
                    row["predicted_class"] == class_name
                    for row in class_rows
                ]))

            per_seed.append(summary)

    metric_names = [
        "accuracy",
        "recall_black_sigatoka",
        "recall_healthy",
        "recall_yellow_sigatoka",
    ]

    mean_sd = []
    for condition in ["original_support", "contrast_augmented_support"]:
        rows = [
            row for row in per_seed
            if row["support_condition"] == condition
        ]
        summary = {"support_condition": condition}

        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd_population"] = float(values.std(ddof=0))

        mean_sd.append(summary)

    return per_seed, mean_sd


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records to write: {path}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Using device:", DEVICE)
    print("Shots per class:", SHOTS)
    print("Seeds:", SEEDS)
    print("Support contrast factor:", CONTRAST_FACTOR)
    print("Test images remain unchanged.")

    # Deterministic support dataset; raw test set stays untouched.
    support_dataset = ImageFolder(
        TRAIN_PATH, transform=validation_transforms
    )
    test_dataset = ImageFolder(
        TEST_PATH, transform=validation_transforms
    )

    if support_dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected support classes: {support_dataset.classes}")
    if test_dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected test classes: {test_dataset.classes}")

    print("Classes:", CLASS_NAMES)
    print("Training images:", len(support_dataset))
    print("Test images:", len(test_dataset))

    model = create_feature_extractor().to(DEVICE)
    model.eval()

    all_records = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        support_indices = get_support_indices(
            support_dataset, SHOTS, seed
        )

        # Condition 1: original support image embeddings.
        original_embeddings, original_labels = extract_embeddings(
            model, support_dataset, support_indices, BATCH_SIZE
        )
        original_prototypes = create_prototypes(
            original_embeddings, original_labels, len(CLASS_NAMES)
        )

        original_rows = evaluate_seed(
            model, test_dataset, original_prototypes, seed, "original_support"
        )
        all_records.extend(original_rows)

        # Condition 2: same selected support image identities, but each image
        # is contrast-enhanced before deterministic preprocessing.
        augmented_dataset = ContrastAugmentedSupport(
            support_dataset, support_indices, CONTRAST_FACTOR
        )
        augmented_indices = list(range(len(augmented_dataset)))
        augmented_embeddings, augmented_labels = extract_embeddings(
            model, augmented_dataset, augmented_indices, BATCH_SIZE
        )
        augmented_prototypes = create_prototypes(
            augmented_embeddings, augmented_labels, len(CLASS_NAMES)
        )

        augmented_rows = evaluate_seed(
            model,
            test_dataset,
            augmented_prototypes,
            seed,
            "contrast_augmented_support",
        )
        all_records.extend(augmented_rows)

    detailed_path = OUTPUT_DIR / "augmented_support_predictions.csv"
    write_csv(detailed_path, all_records)

    per_seed, mean_sd = summarize_records(all_records)
    per_seed_path = OUTPUT_DIR / "per_seed_summary.csv"
    mean_sd_path = OUTPUT_DIR / "mean_sd_summary.csv"

    write_csv(per_seed_path, per_seed)
    write_csv(mean_sd_path, mean_sd)

    print("\nPER-SEED RESULTS")
    print("-" * 105)
    print(
        f"{'Seed':<8}{'Support condition':<32}{'Accuracy':<14}"
        f"{'Black recall':<16}{'Healthy recall':<17}{'Yellow recall':<16}"
    )

    for row in per_seed:
        print(
            f"{row['seed']:<8}{row['support_condition']:<32}"
            f"{row['accuracy'] * 100:>7.2f}%       "
            f"{row['recall_black_sigatoka'] * 100:>7.2f}%        "
            f"{row['recall_healthy'] * 100:>7.2f}%          "
            f"{row['recall_yellow_sigatoka'] * 100:>7.2f}%"
        )

    print("\nMEAN ± POPULATION SD ACROSS SUPPORT SEEDS")
    print("-" * 105)
    for row in mean_sd:
        print(f"\nCondition: {row['support_condition']}")
        for metric in [
            "accuracy",
            "recall_black_sigatoka",
            "recall_healthy",
            "recall_yellow_sigatoka",
        ]:
            print(
                f"  {metric}: {row[f'{metric}_mean'] * 100:.2f}% ± "
                f"{row[f'{metric}_sd_population'] * 100:.2f} percentage points"
            )

    print("\nDetailed predictions:", detailed_path)
    print("Per-seed summary:", per_seed_path)
    print("Mean/SD summary:", mean_sd_path)


if __name__ == "__main__":
    main()