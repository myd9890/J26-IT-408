import csv
import hashlib
from pathlib import Path
from torchvision.datasets import ImageFolder

from src.data.transforms import validation_transforms

import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from src.data.dataset import create_datasets
from src.models.dinov2_feature_extractor import (
    create_dinov2_feature_extractor
)
from src.models.prototypical import (
    create_prototypes,
    classify_with_prototypes
)


# Experiment settings
SHOTS = [1, 5, 10, 20]
SEEDS = [42, 123, 456, 789, 1010]
BATCH_SIZE = 16


def extract_embeddings(model, dataset, device):
    """
    Extract normalized feature embeddings and labels
    from a dataset.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_embeddings = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            embeddings = model(images)

            # L2-normalize embeddings for consistent comparison.
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    return (
        torch.cat(all_embeddings, dim=0),
        torch.cat(all_labels, dim=0)
    )


def get_support_indices(dataset, shots, seed):
    """
    Randomly select the requested number of support images
    from each class.
    """

    random.seed(seed)

    class_indices = {}

    for index, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(index)

    support_indices = []

    for class_id in sorted(class_indices):
        indices = class_indices[class_id]

        if len(indices) < shots:
            raise ValueError(
                f"Class {class_id} has only {len(indices)} "
                f"images, but {shots} shots were requested."
            )

        selected = random.sample(indices, shots)
        support_indices.extend(selected)

    return support_indices


def evaluate():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load the same datasets used in the ResNet18 experiment.
    _, validation_dataset, test_dataset = create_datasets()

    # Create a separate training dataset with deterministic transforms.
    train_dataset = ImageFolder(
        Path("data/processed/train"),
        transform=validation_transforms
    )

    print("\nClasses:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))

    # Load the pretrained DINOv2 feature extractor.
    model = create_dinov2_feature_extractor(device)

    # Extract features once rather than repeatedly for every seed.
    print("\nExtracting training embeddings...")

    train_embeddings, train_labels = extract_embeddings(
        model, train_dataset, device
    )

    print("Extracting test embeddings...")

    test_embeddings, test_labels = extract_embeddings(
        model, test_dataset, device
    )

    # The dataset index corresponds to the embedding row because
    # extraction uses shuffle=False.
    print("\n========== DINOv2 FEW-SHOT RESULTS ==========")

    num_classes = len(train_dataset.classes)

    all_results = []

    for shots in SHOTS:

        accuracies = []

        for seed in SEEDS:

            support_indices = get_support_indices(
                train_dataset, shots, seed
            )

            support_embeddings = train_embeddings[
                support_indices
            ]

            support_labels = train_labels[
                support_indices
            ]

            prototypes = create_prototypes(
                support_embeddings,
                support_labels,
                num_classes
            )

            predictions, _ = classify_with_prototypes(
                test_embeddings,
                prototypes
            )

            accuracy = (
                predictions == test_labels
            ).float().mean().item()

            accuracies.append(accuracy)

            support_signature = hashlib.sha256(
                ",".join(
                    map(str, sorted(support_indices))
                ).encode()
            ).hexdigest()

            all_results.append({
                "model": "DINOv2",
                "shots": shots,
                "seed": seed,
                "accuracy": accuracy,
                "support_signature": support_signature
            })

            print(
                f"{shots}-shot | Seed {seed} | "
                f"Accuracy: {accuracy * 100:.2f}%"
            )

        mean_accuracy = np.mean(accuracies) * 100
        std_accuracy = np.std(accuracies, ddof=0) * 100

        print(
            f"\n{shots}-shot summary: "
            f"{mean_accuracy * 100:.2f}% ± "
            f"{std_accuracy * 100:.2f}%"
        )
        print("-" * 45)

    print("\nExperiment completed.")

    output_path = Path("results/dinov2_few_shot_runs.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "shots",
                "seed",
                "accuracy",
                "support_signature"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    evaluate()

