import random
from pathlib import Path

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

    return (
        torch.cat(embeddings),
        torch.cat(labels),
    )


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

    test_embeddings, test_labels = extract_embeddings(
        model,
        test_dataset,
    )

    distances = torch.cdist(
        test_embeddings,
        prototypes,
    )

    predictions = distances.argmin(dim=1)

    y_true = test_labels.numpy()
    y_pred = predictions.numpy()

    class_names = test_dataset.classes

    print("\n========== 20-SHOT EVALUATION ==========")

    print(
        f"Accuracy: "
        f"{accuracy_score(y_true, y_pred) * 100:.2f}%"
    )

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

    output_dir = Path("results/gradcam_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i in range(len(test_dataset)):

        image_path = test_dataset.samples[i][0]

        true_class = class_names[y_true[i]]
        predicted_class = class_names[y_pred[i]]

        records.append({
            "image_index": i,
            "image_path": image_path,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "correct": y_true[i] == y_pred[i],
            "black_sigatoka_distance": distances[i, 0].item(),
            "healthy_distance": distances[i, 1].item(),
            "yellow_sigatoka_distance": distances[i, 2].item(),
        })

    output_path = output_dir / "all_test_predictions.csv"

    pd.DataFrame(records).to_csv(
        output_path,
        index=False,
    )

    print("\nSaved all test predictions to:", output_path)


if __name__ == "__main__":
    main()