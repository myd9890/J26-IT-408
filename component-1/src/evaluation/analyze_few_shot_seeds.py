import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import recall_score, accuracy_score

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.data.dataset import create_datasets
from src.data.transforms import validation_transforms
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import create_prototypes


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SHOTS = [1, 5, 10, 20]
SEEDS = [42, 123, 456, 789, 1010]
BATCH_SIZE = 16


def get_support_indices(dataset, shots, seed):

    rng = random.Random(seed)
    selected = []

    for class_id in range(len(dataset.classes)):

        indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        selected.extend(rng.sample(indices, shots))

    return selected


def extract_embeddings(model, dataset, indices=None):

    if indices is not None:
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    embeddings, labels = [], []

    model.eval()

    with torch.no_grad():
        for images, batch_labels in loader:

            features = model(images.to(DEVICE))
            features = features.flatten(start_dim=1)

            features = F.normalize(
                features,
                p=2,
                dim=1
            )

            embeddings.append(features.cpu())
            labels.append(batch_labels)

    return torch.cat(embeddings), torch.cat(labels)


def main():

    train_dataset = ImageFolder(
        Path("data/processed/train"),
        transform=validation_transforms
    )

    _, _, test_dataset = create_datasets()

    model = create_feature_extractor().to(DEVICE)
    model.eval()

    test_embeddings, test_labels = extract_embeddings(
        model,
        test_dataset
    )

    results = []

    for shots in SHOTS:

        for seed in SEEDS:

            support_indices = get_support_indices(
                train_dataset,
                shots,
                seed
            )

            support_embeddings, support_labels = (
                extract_embeddings(
                    model,
                    train_dataset,
                    support_indices
                )
            )

            prototypes = create_prototypes(
                support_embeddings,
                support_labels,
                len(train_dataset.classes)
            )

            distances = torch.cdist(
                test_embeddings,
                prototypes
            )

            predictions = distances.argmin(dim=1)

            y_true = test_labels.numpy()
            y_pred = predictions.numpy()

            results.append({
                "model": "ResNet18",
                "shots": shots,
                "seed": seed,
                "accuracy": accuracy_score(y_true, y_pred),
                "black_sigatoka_recall": recall_score(
                    y_true,
                    y_pred,
                    labels=[0],
                    average=None,
                    zero_division=0
                )[0],
                "healthy_recall": recall_score(
                    y_true,
                    y_pred,
                    labels=[1],
                    average=None,
                    zero_division=0
                )[0],
                "yellow_sigatoka_recall": recall_score(
                    y_true,
                    y_pred,
                    labels=[2],
                    average=None,
                    zero_division=0
                )[0]
            })

    output_dir = Path("results/few_shot_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_dir / "resnet18_seed_analysis.csv",
        index=False
    )

    summary = (
        results_df
        .groupby("shots")[
            [
                "accuracy",
                "black_sigatoka_recall",
                "healthy_recall",
                "yellow_sigatoka_recall"
            ]
        ]
        .mean()
        * 100
    )

    print("\n========== FIVE-SEED ANALYSIS ==========")
    print(summary.round(2).to_string())

    print(
        "\nDetailed results saved to "
        "results/few_shot_analysis/resnet18_seed_analysis.csv"
    )


if __name__ == "__main__":
    main()