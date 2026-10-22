import random
import statistics
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.data.dataset import create_datasets
from src.data.transforms import validation_transforms
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import create_prototypes


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SHOTS = [1, 5, 10, 20]

SEEDS = [42, 123, 456, 789, 1010]

BATCH_SIZE = 16


# --------------------------------------------------
# SUPPORT SET SELECTION
# --------------------------------------------------

def get_support_indices(dataset, shots, seed):

    rng = random.Random(seed)

    support_indices = []

    for class_id in range(len(dataset.classes)):

        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        selected = rng.sample(
            class_indices,
            shots
        )

        support_indices.extend(selected)

    return support_indices


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_embeddings(model, dataset, indices=None):

    if indices is not None:
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
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

            embeddings.append(features.cpu())
            labels.append(batch_labels)

    return (
        torch.cat(embeddings),
        torch.cat(labels)
    )


# --------------------------------------------------
# MEAN PROTOTYPES
# --------------------------------------------------

def create_mean_prototypes(
    embeddings,
    labels,
    num_classes
):

    # Uses your existing prototype implementation.
    return create_prototypes(
        embeddings,
        labels,
        num_classes
    )


# --------------------------------------------------
# MEDOID PROTOTYPES
# --------------------------------------------------

def create_medoid_prototypes(
    embeddings,
    labels,
    num_classes
):

    prototypes = []

    for class_id in range(num_classes):

        class_embeddings = embeddings[
            labels == class_id
        ]

        # Calculate pairwise Euclidean distances between
        # all support embeddings belonging to this class.
        pairwise_distances = torch.cdist(
            class_embeddings,
            class_embeddings,
            p=2
        )

        # Find the support example with the smallest
        # total distance to the other examples.
        average_distances = pairwise_distances.mean(
            dim=1
        )

        medoid_index = average_distances.argmin()

        medoid = class_embeddings[medoid_index]

        prototypes.append(medoid)

    return torch.stack(prototypes)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(
    test_embeddings,
    test_labels,
    prototypes
):

    distances = torch.cdist(
        test_embeddings,
        prototypes,
        p=2
    )

    predictions = distances.argmin(
        dim=1
    )

    y_true = test_labels.numpy()
    y_pred = predictions.numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    class_recalls = recall_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "black_sigatoka_recall": class_recalls[0],
        "healthy_recall": class_recalls[1],
        "yellow_sigatoka_recall": class_recalls[2]
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    train_dataset = ImageFolder(
        Path("data/processed/train"),
        transform=validation_transforms
    )

    _, _, test_dataset = create_datasets()

    print("\nClasses:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    model = create_feature_extractor().to(DEVICE)
    model.eval()

    # Extract test embeddings only once.
    print("\nExtracting test embeddings...")

    test_embeddings, test_labels = extract_embeddings(
        model,
        test_dataset
    )

    records = []

    for shots in SHOTS:

        print(f"\n========== {shots}-SHOT ==========")

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

            # Mean prototype method.
            mean_prototypes = create_mean_prototypes(
                support_embeddings,
                support_labels,
                len(train_dataset.classes)
            )

            mean_results = evaluate(
                test_embeddings,
                test_labels,
                mean_prototypes
            )

            records.append({
                "method": "mean",
                "shots": shots,
                "seed": seed,
                **mean_results
            })

            # Medoid prototype method.
            medoid_prototypes = create_medoid_prototypes(
                support_embeddings,
                support_labels,
                len(train_dataset.classes)
            )

            medoid_results = evaluate(
                test_embeddings,
                test_labels,
                medoid_prototypes
            )

            records.append({
                "method": "medoid",
                "shots": shots,
                "seed": seed,
                **medoid_results
            })

            print(
                f"Seed {seed} | "
                f"Mean accuracy: "
                f"{mean_results['accuracy'] * 100:.2f}% | "
                f"Medoid accuracy: "
                f"{medoid_results['accuracy'] * 100:.2f}%"
            )

    # --------------------------------------------------
    # SAVE DETAILED RESULTS
    # --------------------------------------------------

    output_dir = Path(
        "results/prototype_method_comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(records)

    results_df.to_csv(
        output_dir / "individual_results.csv",
        index=False
    )

    # --------------------------------------------------
    # CALCULATE MEAN AND STANDARD DEVIATION
    # --------------------------------------------------

    metrics = [
        "accuracy",
        "black_sigatoka_recall",
        "healthy_recall",
        "yellow_sigatoka_recall"
    ]

    summary_records = []

    for (method, shots), group in results_df.groupby(
        ["method", "shots"]
    ):

        summary = {
            "method": method,
            "shots": shots
        }

        for metric in metrics:

            values = group[metric].tolist()

            summary[f"{metric}_mean"] = (
                statistics.mean(values) * 100
            )

            summary[f"{metric}_std"] = (
                statistics.stdev(values) * 100
            )

        summary_records.append(summary)

    summary_df = pd.DataFrame(
        summary_records
    ).sort_values(
        ["shots", "method"]
    )

    summary_df.to_csv(
        output_dir / "summary_results.csv",
        index=False
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n========== SUMMARY RESULTS ==========")

    display_columns = [
        "method",
        "shots",
        "accuracy_mean",
        "accuracy_std",
        "black_sigatoka_recall_mean",
        "black_sigatoka_recall_std",
        "healthy_recall_mean",
        "yellow_sigatoka_recall_mean"
    ]

    print(
        summary_df[display_columns]
        .round(2)
        .to_string(index=False)
    )

    print("\nResults saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()