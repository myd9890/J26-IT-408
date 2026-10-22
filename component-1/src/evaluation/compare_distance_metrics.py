import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.data.dataset import create_datasets
from src.data.transforms import validation_transforms
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import create_prototypes


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SHOTS = 20
SEED = 42
BATCH_SIZE = 16


def get_support_indices(dataset, shots, seed):

    rng = random.Random(seed)
    support_indices = []

    for class_id in range(len(dataset.classes)):

        class_indices = [
            i
            for i, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        support_indices.extend(
            rng.sample(class_indices, shots)
        )

    return support_indices


def extract_embeddings(model, dataset, indices=None):

    if indices is not None:
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    embeddings = []
    labels = []

    model.eval()

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(DEVICE)

            features = model(images)
            features = features.flatten(start_dim=1)

            features = F.normalize(
                features,
                p=2,
                dim=1,
            )

            embeddings.append(features.cpu())
            labels.append(batch_labels)

    return torch.cat(embeddings), torch.cat(labels)


def evaluate_metric(
    metric_name,
    test_embeddings,
    test_labels,
    prototypes,
    class_names,
):

    if metric_name == "euclidean":

        distances = torch.cdist(
            test_embeddings,
            prototypes,
            p=2,
        )

        predictions = distances.argmin(dim=1)

    elif metric_name == "cosine":

        # Prototypes are already normalized after construction
        # in the create_prototypes implementation below.
        normalized_prototypes = F.normalize(
            prototypes,
            p=2,
            dim=1,
        )

        similarities = (
            test_embeddings @ normalized_prototypes.T
        )

        predictions = similarities.argmax(dim=1)

    else:
        raise ValueError(
            f"Unsupported metric: {metric_name}"
        )

    y_true = test_labels.numpy()
    y_pred = predictions.numpy()

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n========== {metric_name.upper()} ==========")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
        )
    )

    return {
        "metric": metric_name,
        "accuracy": accuracy,
        "predictions": y_pred,
    }


def main():

    print("Using device:", DEVICE)

    train_dataset = ImageFolder(
        Path("data/processed/train"),
        transform=validation_transforms,
    )

    _, _, test_dataset = create_datasets()

    model = create_feature_extractor().to(DEVICE)
    model.eval()

    support_indices = get_support_indices(
        train_dataset,
        SHOTS,
        SEED,
    )

    support_embeddings, support_labels = extract_embeddings(
        model,
        train_dataset,
        support_indices,
    )

    prototypes = create_prototypes(
        support_embeddings,
        support_labels,
        len(train_dataset.classes),
    )

    # Normalize prototypes explicitly for cosine similarity.
    prototypes = F.normalize(
        prototypes,
        p=2,
        dim=1,
    )

    test_embeddings, test_labels = extract_embeddings(
        model,
        test_dataset,
    )

    results = []

    for metric_name in ["euclidean", "cosine"]:

        result = evaluate_metric(
            metric_name,
            test_embeddings,
            test_labels,
            prototypes,
            test_dataset.classes,
        )

        results.append(result)

    output_dir = Path("results/distance_metric_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([
        {
            "metric": result["metric"],
            "accuracy": result["accuracy"],
        }
        for result in results
    ])

    summary.to_csv(
        output_dir / "metric_summary.csv",
        index=False,
    )

    predictions_df = pd.DataFrame({
        "image_index": range(len(test_dataset)),
        "true_class": [
            test_dataset.classes[label]
            for label in test_labels.tolist()
        ],
        "euclidean_prediction": [
            test_dataset.classes[label]
            for label in results[0]["predictions"]
        ],
        "cosine_prediction": [
            test_dataset.classes[label]
            for label in results[1]["predictions"]
        ],
    })

    predictions_df.to_csv(
        output_dir / "metric_predictions.csv",
        index=False,
    )

    print("\nSaved results to:", output_dir)


if __name__ == "__main__":
    main()