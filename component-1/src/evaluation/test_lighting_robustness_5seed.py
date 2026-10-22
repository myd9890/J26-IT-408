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
from src.data.dataset import create_datasets
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import create_prototypes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHOTS = 20
SEEDS = [42, 123, 456, 789, 1010]
BATCH_SIZE = 16
TRAIN_PATH = Path("data/processed/train")
OUTPUT_DIR = Path("results/lighting_robustness_5seed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["black_sigatoka", "healthy", "yellow_sigatoka"]
CONDITIONS = {"original": 1.0, "higher_contrast": 1.30}


def get_support_indices(dataset, shots, seed):
    # Match the exact sampling logic used in evaluate_few_shot.py.
    random.seed(seed)
    support_indices = []
    for class_id in range(len(dataset.classes)):
        class_indices = [
            index for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]
        support_indices.extend(random.sample(class_indices, shots))
    return support_indices


def extract_embeddings(model, dataset, indices, batch_size=16):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            features = model(images.to(DEVICE)).flatten(start_dim=1)
            features = F.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu())
            labels.append(batch_labels)
    return torch.cat(embeddings), torch.cat(labels)


def transform_image(image, condition):
    if condition == "original":
        return image.copy()
    if condition == "higher_contrast":
        return ImageEnhance.Contrast(image.copy()).enhance(1.30)
    raise ValueError(f"Unknown condition: {condition}")


def predict_tensor(model, image_tensor, prototypes):
    model.eval()
    with torch.no_grad():
        embedding = model(image_tensor.unsqueeze(0).to(DEVICE))
        embedding = F.normalize(embedding.flatten(start_dim=1), p=2, dim=1)
        distances = torch.cdist(embedding.cpu(), prototypes.cpu(), p=2)[0]
        predicted_id = int(distances.argmin().item())
    return predicted_id, distances.numpy()


def main():
    print("Using device:", DEVICE)
    train_dataset = ImageFolder(TRAIN_PATH, transform=validation_transforms)
    _, _, test_dataset = create_datasets()

    if train_dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected train classes: {train_dataset.classes}")
    if test_dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected test classes: {test_dataset.classes}")

    print("Classes:", CLASS_NAMES)
    print("Test images:", len(test_dataset))
    print("Seeds:", SEEDS)
    print("Conditions:", list(CONDITIONS))

    test_images = []
    for image_path, actual_id in test_dataset.samples:
        image = Image.open(image_path).convert("RGB")
        test_images.append((Path(image_path), actual_id, image))

    all_records = []

    for seed in SEEDS:
        print(f"\n--- Support seed {seed} ---")
        support_indices = get_support_indices(train_dataset, SHOTS, seed)
        model = create_feature_extractor().to(DEVICE)
        model.eval()

        support_embeddings, support_labels = extract_embeddings(
            model, train_dataset, support_indices, BATCH_SIZE
        )
        prototypes = create_prototypes(
            support_embeddings, support_labels, len(CLASS_NAMES)
        )

        for image_path, actual_id, original_image in test_images:
            for condition in CONDITIONS:
                transformed_image = transform_image(original_image, condition)
                image_tensor = validation_transforms(transformed_image)
                predicted_id, distances = predict_tensor(
                    model, image_tensor, prototypes
                )
                all_records.append({
                    "seed": seed,
                    "shots_per_class": SHOTS,
                    "image": image_path.name,
                    "actual_class": CLASS_NAMES[actual_id],
                    "condition": condition,
                    "predicted_class": CLASS_NAMES[predicted_id],
                    "correct": predicted_id == actual_id,
                    "distance_black_sigatoka": float(distances[0]),
                    "distance_healthy": float(distances[1]),
                    "distance_yellow_sigatoka": float(distances[2]),
                })

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    detailed_path = OUTPUT_DIR / "five_seed_lighting_predictions.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)

    per_seed = []
    for seed in SEEDS:
        seed_records = [r for r in all_records if r["seed"] == seed]
        for condition in CONDITIONS:
            rows = [r for r in seed_records if r["condition"] == condition]
            summary = {
                "seed": seed,
                "condition": condition,
                "accuracy": float(np.mean([r["correct"] for r in rows])),
            }
            for class_name in CLASS_NAMES:
                class_rows = [r for r in rows if r["actual_class"] == class_name]
                summary[f"recall_{class_name}"] = float(np.mean([
                    r["predicted_class"] == class_name for r in class_rows
                ]))
            per_seed.append(summary)

    per_seed_path = OUTPUT_DIR / "per_seed_summary.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_seed[0].keys())
        writer.writeheader()
        writer.writerows(per_seed)

    metrics = [
        "accuracy", "recall_black_sigatoka",
        "recall_healthy", "recall_yellow_sigatoka"
    ]
    summary_records = []
    for condition in CONDITIONS:
        rows = [r for r in per_seed if r["condition"] == condition]
        summary = {"condition": condition}
        for metric in metrics:
            values = np.array([r[metric] for r in rows], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd_population"] = float(values.std(ddof=0))
        summary_records.append(summary)

    summary_path = OUTPUT_DIR / "mean_sd_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_records[0].keys())
        writer.writeheader()
        writer.writerows(summary_records)

    print("\nPER-SEED RESULTS")
    print("-" * 100)
    print(
        f"{'Seed':<8}{'Condition':<20}{'Accuracy':<14}"
        f"{'Black recall':<16}{'Healthy recall':<17}{'Yellow recall':<16}"
    )
    for row in per_seed:
        print(
            f"{row['seed']:<8}{row['condition']:<20}"
            f"{row['accuracy'] * 100:>7.2f}%       "
            f"{row['recall_black_sigatoka'] * 100:>7.2f}%        "
            f"{row['recall_healthy'] * 100:>7.2f}%          "
            f"{row['recall_yellow_sigatoka'] * 100:>7.2f}%"
        )

    print("\nMEAN ± POPULATION SD ACROSS SUPPORT SEEDS")
    print("-" * 100)
    for row in summary_records:
        print(f"\nCondition: {row['condition']}")
        for metric in metrics:
            print(
                f"  {metric}: {row[f'{metric}_mean'] * 100:.2f}% ± "
                f"{row[f'{metric}_sd_population'] * 100:.2f} percentage points"
            )

    print("\nDetailed predictions:", detailed_path)
    print("Per-seed summary:", per_seed_path)
    print("Mean/SD summary:", summary_path)


if __name__ == "__main__":
    main()