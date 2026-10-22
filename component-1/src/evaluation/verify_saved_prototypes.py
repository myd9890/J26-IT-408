import random
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.data.transforms import validation_transforms
from src.data.dataset import create_datasets

from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import (
    create_prototypes,
    classify_with_prototypes,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_DIR = Path("data/processed/train")

PROTOTYPE_PATH = Path(
    "models/resnet18_prototypes_20shot_seed42.pth"
)

SHOTS = 20
SEED = 42
BATCH_SIZE = 16


def get_support_indices(dataset, shots, seed):

    random.seed(seed)

    support_indices = []

    for class_id in range(len(dataset.classes)):

        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        selected = random.sample(
            class_indices,
            shots
        )

        support_indices.extend(selected)

    return support_indices


def extract_embeddings(model, dataset, indices):

    subset = torch.utils.data.Subset(
        dataset,
        indices
    )

    loader = DataLoader(
        subset,
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

    print("Device:", DEVICE)

    # Load datasets using deterministic transforms.
    train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=validation_transforms
    )

    _, _, test_dataset = create_datasets()

    print("Classes:", train_dataset.classes)
    print("Test images:", len(test_dataset))

    # Reconstruct the support set used in the original evaluator.
    support_indices = get_support_indices(
        train_dataset,
        SHOTS,
        SEED
    )

    support_signature = hashlib.sha256(
        ",".join(
            map(str, sorted(support_indices))
        ).encode()
    ).hexdigest()

    print("\nReconstructed support signature:")
    print(support_signature)

    # Load saved artifact.
    # weights_only=False is needed because this artifact
    # contains metadata as well as tensors.
    artifact = torch.load(
        PROTOTYPE_PATH,
        map_location="cpu",
        weights_only=False
    )

    print("\nSaved class names:")
    print(artifact["class_names"])

    print("Saved support signature:")
    print(artifact["support_signature"])

    assert support_signature == artifact["support_signature"], (
        "Support-set signature mismatch!"
    )

    assert train_dataset.classes == artifact["class_names"], (
        "Class ordering mismatch!"
    )

    # Reconstruct prototypes independently.
    feature_extractor = (
        create_feature_extractor()
        .to(DEVICE)
    )

    support_embeddings, support_labels = extract_embeddings(
        feature_extractor,
        train_dataset,
        support_indices
    )

    reconstructed_prototypes = create_prototypes(
        support_embeddings,
        support_labels,
        len(train_dataset.classes)
    )

    saved_prototypes = artifact["prototypes"]

    max_difference = (
        reconstructed_prototypes - saved_prototypes
    ).abs().max().item()

    print("\nMaximum prototype difference:")
    print(f"{max_difference:.10f}")

    # Calculate test embeddings once.
    test_indices = list(range(len(test_dataset)))

    test_embeddings, test_labels = extract_embeddings(
        feature_extractor,
        test_dataset,
        test_indices
    )

    # Predictions from the reconstructed prototypes.
    reconstructed_predictions, _ = classify_with_prototypes(
        test_embeddings,
        reconstructed_prototypes
    )

    # Predictions from the saved prototypes.
    saved_predictions, saved_distances = classify_with_prototypes(
        test_embeddings,
        saved_prototypes
    )

    matches = (
        reconstructed_predictions == saved_predictions
    )

    matching_predictions = matches.sum().item()
    total_predictions = len(test_dataset)

    print("\n========== VERIFICATION RESULTS ==========")

    print(
        f"Matching predictions: "
        f"{matching_predictions}/{total_predictions}"
    )

    print(
        f"Prediction agreement: "
        f"{100 * matching_predictions / total_predictions:.2f}%"
    )

    print(
        "Saved-prototype test accuracy:",
        f"{100 * (saved_predictions == test_labels).float().mean().item():.2f}%"
    )

    print("Distance matrix shape:", tuple(saved_distances.shape))

    if matching_predictions == total_predictions:
        print("\nSUCCESS: Saved prototypes reproduce the reconstructed predictions.")
    else:
        print("\nWARNING: Some predictions differ.")

    if max_difference < 1e-6:
        print("SUCCESS: Saved prototypes match reconstructed prototypes.")
    else:
        print("WARNING: Prototype values differ beyond the tolerance.")


if __name__ == "__main__":
    main()