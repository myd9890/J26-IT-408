import csv
import random
import hashlib
from pathlib import Path

import numpy as np

from torchvision.datasets import ImageFolder

from src.data.transforms import validation_transforms

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from src.data.dataset import create_datasets
from src.models.feature_extractor import (
    create_feature_extractor
)
from src.models.prototypical import (
    create_prototypes,
    classify_with_prototypes,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SHOTS = [1, 5, 10, 20]

SEEDS = [
    42,
    123,
    456,
    789,
    1010,
]


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

            # L2 normalization
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


def get_support_indices(
    dataset,
    shots,
    seed
):

    random.seed(seed)

    support_indices = []

    for class_id in range(
        len(dataset.classes)
    ):

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


def main():

    print("Using device:", DEVICE)

    _, _, test_dataset = create_datasets()

    # Use deterministic preprocessing for few-shot support images.
    train_dataset = ImageFolder(
        Path("data/processed/train"),
        transform=validation_transforms
    )

    feature_extractor = (
        create_feature_extractor()
        .to(DEVICE)
    )

    print("\nClasses:")
    print(train_dataset.classes)

    # Extract test embeddings once
    test_indices = list(
        range(len(test_dataset))
    )

    test_embeddings, test_labels = (
        extract_embeddings(
            feature_extractor,
            test_dataset,
            test_indices
        )
    )

    print(
        "\nTest images:",
        len(test_dataset)
    )

    all_results = []

    print("\n========== FEW-SHOT RESULTS ==========")

    for shots in SHOTS:

        accuracies = []

        for seed in SEEDS:

            support_indices = get_support_indices(
                train_dataset,
                shots,
                seed
            )

            support_embeddings, support_labels = (
                extract_embeddings(
                    feature_extractor,
                    train_dataset,
                    support_indices
                )
            )

            prototypes = create_prototypes(
                support_embeddings,
                support_labels,
                len(train_dataset.classes)
            )

            predictions, _ = (
                classify_with_prototypes(
                    test_embeddings,
                    prototypes
                )
            )

            accuracy = (
                predictions == test_labels
            ).float().mean().item()

            accuracies.append(accuracy)

            # Create a signature to verify support-set consistency.
            support_signature = hashlib.sha256(
                ",".join(
                    map(str, sorted(support_indices))
                ).encode()
            ).hexdigest()

            all_results.append({
                "model": "ResNet18",
                "shots": shots,
                "seed": seed,
                "accuracy": accuracy,
                "support_signature": support_signature
            })

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies, ddof=0)

        print(
            f"{shots}-shot: "
            f"{mean_accuracy * 100:.2f}% ± "
            f"{std_accuracy * 100:.2f}%"
        )

    output_path = Path("results/resnet18_few_shot_runs.csv")
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
    main()