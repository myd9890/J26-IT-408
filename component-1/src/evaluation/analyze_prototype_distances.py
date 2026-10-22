import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.data.transforms import validation_transforms
from src.data.dataset import create_datasets
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import create_prototypes


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SHOTS = 20
SEED = 42
BATCH_SIZE = 16

TRAIN_PATH = Path("data/processed/train")

OUTPUT_DIR = Path(
    "results/prototype_distance_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# SUPPORT SELECTION
# Copied from evaluate_few_shot.py
# --------------------------------------------------

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


# --------------------------------------------------
# EMBEDDING EXTRACTION
# Copied from evaluate_few_shot.py
# --------------------------------------------------

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

    all_embeddings = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

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

            all_embeddings.append(
                features.cpu()
            )

            all_labels.append(labels)

    return (
        torch.cat(all_embeddings),
        torch.cat(all_labels)
    )


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    # Same datasets and deterministic support preprocessing
    _, _, test_dataset = create_datasets()

    train_dataset = ImageFolder(
        TRAIN_PATH,
        transform=validation_transforms
    )

    class_names = train_dataset.classes

    print("\nClasses:", class_names)
    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    # Use the same feature extractor as the existing evaluation
    feature_extractor = (
        create_feature_extractor()
        .to(DEVICE)
    )

    feature_extractor.eval()

    # Select the exact support set for 20 shots, seed 42
    support_indices = get_support_indices(
        train_dataset,
        SHOTS,
        SEED
    )

    print("\nSupport images:", len(support_indices))
    print("Support seed:", SEED)
    print("Shots per class:", SHOTS)

    # Extract support embeddings
    support_embeddings, support_labels = (
        extract_embeddings(
            feature_extractor,
            train_dataset,
            support_indices,
            BATCH_SIZE
        )
    )

    # Construct mean prototypes
    prototypes = create_prototypes(
        support_embeddings,
        support_labels,
        len(class_names)
    )

    # Extract test embeddings using the existing test dataset
    test_indices = list(range(len(test_dataset)))

    test_embeddings, test_labels = (
        extract_embeddings(
            feature_extractor,
            test_dataset,
            test_indices,
            BATCH_SIZE
        )
    )

    # Calculate Euclidean distances
    distances = torch.cdist(
        test_embeddings,
        prototypes,
        p=2
    ).numpy()

    actual_labels = test_labels.numpy()

    predicted_labels = np.argmin(
        distances,
        axis=1
    )

    # --------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------

    records = []

    for index in range(len(test_dataset)):

        actual_id = int(actual_labels[index])
        predicted_id = int(predicted_labels[index])

        sorted_distances = np.sort(
            distances[index]
        )

        nearest_distance = sorted_distances[0]
        second_nearest_distance = sorted_distances[1]

        image_path, _ = test_dataset.samples[index]

        records.append({
            "image": Path(image_path).name,
            "actual_class": class_names[actual_id],
            "predicted_class": class_names[predicted_id],
            "correct": actual_id == predicted_id,
            "distance_black_sigatoka": distances[index, 0],
            "distance_healthy": distances[index, 1],
            "distance_yellow_sigatoka": distances[index, 2],
            "nearest_distance": nearest_distance,
            "second_nearest_distance": second_nearest_distance,
            "distance_margin": (
                second_nearest_distance - nearest_distance
            )
        })

    # --------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------

    output_path = (
        OUTPUT_DIR / "prototype_distances.csv"
    )

    with output_path.open(
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

    # --------------------------------------------------
    # BLACK SIGATOKA ANALYSIS
    # --------------------------------------------------

    black_results = [
        row
        for row in records
        if row["actual_class"] == "black_sigatoka"
    ]

    print("\n" + "=" * 110)
    print("BLACK SIGATOKA TEST IMAGES")
    print("=" * 110)

    print(
        f"{'Image':<25}"
        f"{'Prediction':<20}"
        f"{'Correct':<10}"
        f"{'D(Black)':<14}"
        f"{'D(Healthy)':<14}"
        f"{'D(Yellow)':<14}"
        f"{'Margin':<12}"
    )

    for row in black_results:

        print(
            f"{row['image'][:24]:<25}"
            f"{row['predicted_class']:<20}"
            f"{str(row['correct']):<10}"
            f"{row['distance_black_sigatoka']:<14.4f}"
            f"{row['distance_healthy']:<14.4f}"
            f"{row['distance_yellow_sigatoka']:<14.4f}"
            f"{row['distance_margin']:<12.4f}"
        )

    # --------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------

    accuracy = np.mean([
        row["correct"]
        for row in records
    ]) * 100

    print("\n" + "=" * 70)
    print("OVERALL CLASSIFICATION SUMMARY")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy:.2f}% "
        f"({sum(row['correct'] for row in records)}/{len(records)})"
    )

    print("\nAccuracy by actual class:")

    for class_name in class_names:

        class_records = [
            row
            for row in records
            if row["actual_class"] == class_name
        ]

        class_accuracy = np.mean([
            row["correct"]
            for row in class_records
        ]) * 100

        print(
            f"{class_name:<20}"
            f"{class_accuracy:>8.2f}% "
            f"({sum(row['correct'] for row in class_records)}"
            f"/{len(class_records)})"
        )

    # --------------------------------------------------
    # DISTANCE SUMMARY
    # --------------------------------------------------

    print("\nMean distances grouped by actual class:")

    distance_columns = [
        "distance_black_sigatoka",
        "distance_healthy",
        "distance_yellow_sigatoka"
    ]

    print(
        f"{'Actual class':<20}"
        f"{'To Black':<15}"
        f"{'To Healthy':<15}"
        f"{'To Yellow':<15}"
    )

    for class_name in class_names:

        class_records = [
            row
            for row in records
            if row["actual_class"] == class_name
        ]

        means = [
            np.mean([
                row[column]
                for row in class_records
            ])
            for column in distance_columns
        ]

        print(
            f"{class_name:<20}"
            f"{means[0]:<15.4f}"
            f"{means[1]:<15.4f}"
            f"{means[2]:<15.4f}"
        )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()