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


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
SHOTS = 20
BATCH_SIZE = 16

TRAIN_PATH = Path("data/processed/train")
OUTPUT_DIR = Path("results/lighting_robustness")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLASS_NAMES = [
    "black_sigatoka",
    "healthy",
    "yellow_sigatoka",
]

# Controlled transformations.
CONDITIONS = {
    "original": (1.0, 1.0),
    "darker": (0.70, 1.0),
    "brighter": (1.30, 1.0),
    "lower_contrast": (1.0, 0.70),
    "higher_contrast": (1.0, 1.30),
}


# ============================================================
# SUPPORT SET SELECTION
# Same procedure as evaluate_few_shot.py
# ============================================================

def get_support_indices(dataset, shots, seed):

    random.seed(seed)

    support_indices = []

    for class_id in range(len(dataset.classes)):

        class_indices = [
            index
            for index, (_, label)
            in enumerate(dataset.samples)
            if label == class_id
        ]

        selected = random.sample(
            class_indices,
            shots
        )

        support_indices.extend(selected)

    return support_indices


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_embeddings(
    model,
    dataset,
    indices,
    batch_size=16
):

    subset = torch.utils.data.Subset(
        dataset,
        indices
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    embeddings = []
    labels = []

    model.eval()

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(DEVICE)

            features = model(images)

            features = features.flatten(
                start_dim=1
            )

            features = F.normalize(
                features,
                p=2,
                dim=1
            )

            embeddings.append(
                features.cpu()
            )

            labels.append(
                batch_labels
            )

    return (
        torch.cat(embeddings),
        torch.cat(labels)
    )


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

def transform_image(image, condition):

    brightness_factor, contrast_factor = (
        CONDITIONS[condition]
    )

    transformed = image.copy()

    if brightness_factor != 1.0:

        transformed = ImageEnhance.Brightness(
            transformed
        ).enhance(
            brightness_factor
        )

    if contrast_factor != 1.0:

        transformed = ImageEnhance.Contrast(
            transformed
        ).enhance(
            contrast_factor
        )

    return transformed


def preprocess_image(image):

    # Apply the same deterministic preprocessing used by
    # the original validation/test dataset.
    return validation_transforms(image)


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    image_tensor,
    prototypes
):

    model.eval()

    with torch.no_grad():

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        embedding = model(
            image_tensor
        )

        embedding = embedding.flatten(
            start_dim=1
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        distances = torch.cdist(
            embedding.cpu(),
            prototypes.cpu(),
            p=2
        )[0]

        predicted_id = int(
            distances.argmin().item()
        )

    sorted_distances = torch.sort(
        distances
    ).values

    margin = float(
        sorted_distances[1] -
        sorted_distances[0]
    )

    return (
        predicted_id,
        distances.numpy(),
        margin
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Using device:", DEVICE)

    # Deterministic support images.
    train_dataset = ImageFolder(
        TRAIN_PATH,
        transform=validation_transforms
    )

    # Original test dataset and labels.
    _, _, test_dataset = create_datasets()

    if train_dataset.classes != CLASS_NAMES:

        raise ValueError(
            f"Unexpected class ordering: "
            f"{train_dataset.classes}"
        )

    if test_dataset.classes != CLASS_NAMES:

        raise ValueError(
            f"Unexpected test class ordering: "
            f"{test_dataset.classes}"
        )

    print("\nClasses:", CLASS_NAMES)
    print("Test images:", len(test_dataset))

    # --------------------------------------------------------
    # Construct the same 20-shot seed-42 prototypes.
    # --------------------------------------------------------

    support_indices = get_support_indices(
        train_dataset,
        SHOTS,
        SEED
    )

    print(
        "Support images:",
        len(support_indices)
    )

    model = create_feature_extractor().to(
        DEVICE
    )

    model.eval()

    support_embeddings, support_labels = (
        extract_embeddings(
            model,
            train_dataset,
            support_indices,
            BATCH_SIZE
        )
    )

    prototypes = create_prototypes(
        support_embeddings,
        support_labels,
        len(CLASS_NAMES)
    )

    print("Prototypes constructed.")

    # --------------------------------------------------------
    # Evaluate original and transformed test images.
    # --------------------------------------------------------

    records = []

    for index, (image_path, actual_id) in enumerate(
        test_dataset.samples
    ):

        image_path = Path(image_path)

        original_image = Image.open(
            image_path
        ).convert("RGB")

        for condition in CONDITIONS:

            transformed_image = transform_image(
                original_image,
                condition
            )

            image_tensor = preprocess_image(
                transformed_image
            )

            predicted_id, distances, margin = (
                predict_image(
                    model,
                    image_tensor,
                    prototypes
                )
            )

            correct = (
                predicted_id == actual_id
            )

            records.append({
                "image": image_path.name,
                "actual_class": CLASS_NAMES[actual_id],
                "condition": condition,
                "predicted_class": CLASS_NAMES[predicted_id],
                "correct": correct,
                "distance_black_sigatoka": float(distances[0]),
                "distance_healthy": float(distances[1]),
                "distance_yellow_sigatoka": float(distances[2]),
                "distance_margin": margin,
            })

    # --------------------------------------------------------
    # Save individual predictions.
    # --------------------------------------------------------

    detailed_path = (
        OUTPUT_DIR / "lighting_predictions.csv"
    )

    with detailed_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    # --------------------------------------------------------
    # Overall accuracy by lighting condition.
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("OVERALL LIGHTING ROBUSTNESS RESULTS")
    print("=" * 90)

    print(
        f"{'Condition':<20}"
        f"{'Accuracy':<15}"
        f"{'Correct':<15}"
        f"{'Changed predictions':<22}"
    )

    original_predictions = {
        row["image"]: row["predicted_class"]
        for row in records
        if row["condition"] == "original"
    }

    for condition in CONDITIONS:

        condition_records = [
            row
            for row in records
            if row["condition"] == condition
        ]

        accuracy = np.mean([
            row["correct"]
            for row in condition_records
        ]) * 100

        correct_count = sum(
            row["correct"]
            for row in condition_records
        )

        changed = sum(
            row["predicted_class"]
            != original_predictions[row["image"]]
            for row in condition_records
        )

        print(
            f"{condition:<20}"
            f"{accuracy:>7.2f}%       "
            f"{correct_count}/{len(condition_records):<12}"
            f"{changed}/{len(condition_records)}"
        )

    # --------------------------------------------------------
    # Per-class recall by condition.
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("PER-CLASS RECALL")
    print("=" * 90)

    print(
        f"{'Condition':<20}"
        f"{'Black Sigatoka':<20}"
        f"{'Healthy':<20}"
        f"{'Yellow Sigatoka':<20}"
    )

    for condition in CONDITIONS:

        condition_records = [
            row
            for row in records
            if row["condition"] == condition
        ]

        recalls = []

        for class_name in CLASS_NAMES:

            class_records = [
                row
                for row in condition_records
                if row["actual_class"] == class_name
            ]

            recall = np.mean([
                row["predicted_class"] == class_name
                for row in class_records
            ]) * 100

            recalls.append(recall)

        print(
            f"{condition:<20}"
            f"{recalls[0]:>7.2f}%             "
            f"{recalls[1]:>7.2f}%             "
            f"{recalls[2]:>7.2f}%"
        )

    # --------------------------------------------------------
    # Detailed results for the five previously misclassified
    # Black Sigatoka images.
    # --------------------------------------------------------

    target_images = {
        "00011.jpg",
        "00012.jpg",
        "00013.jpg",
        "00015.jpg",
        "00016.jpg",
    }

    print("\n" + "=" * 110)
    print("DETAILED RESULTS: PREVIOUSLY MISCLASSIFIED IMAGES")
    print("=" * 110)

    print(
        f"{'Image':<15}"
        f"{'Condition':<20}"
        f"{'Prediction':<20}"
        f"{'D(Black)':<14}"
        f"{'D(Healthy)':<14}"
        f"{'D(Yellow)':<14}"
    )

    for row in records:

        if row["image"] not in target_images:
            continue

        print(
            f"{row['image']:<15}"
            f"{row['condition']:<20}"
            f"{row['predicted_class']:<20}"
            f"{row['distance_black_sigatoka']:<14.4f}"
            f"{row['distance_healthy']:<14.4f}"
            f"{row['distance_yellow_sigatoka']:<14.4f}"
        )

    print("\nDetailed predictions saved to:")
    print(detailed_path)


if __name__ == "__main__":

    main()