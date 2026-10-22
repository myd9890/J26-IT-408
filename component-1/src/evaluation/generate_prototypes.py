import csv
import random
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.data.transforms import validation_transforms
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import create_prototypes


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_DIR = Path("data/processed/train")

OUTPUT_DIR = Path("models")

PROTOTYPE_PATH = (
    OUTPUT_DIR / "resnet18_prototypes_20shot_seed42.pth"
)

SUPPORT_CSV_PATH = (
    OUTPUT_DIR / "resnet18_prototypes_20shot_seed42_support.csv"
)

SHOTS = 20
SEED = 42
BATCH_SIZE = 16

EXPECTED_CLASSES = [
    "black_sigatoka",
    "healthy",
    "yellow_sigatoka",
]


# --------------------------------------------------
# Support image selection
# Must match evaluate_few_shot.py
# --------------------------------------------------

def get_support_indices(dataset, shots, seed):

    random.seed(seed)

    support_indices = []

    for class_id in range(len(dataset.classes)):

        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        if len(class_indices) < shots:
            raise ValueError(
                f"Class '{dataset.classes[class_id]}' "
                f"has only {len(class_indices)} images."
            )

        selected = random.sample(
            class_indices,
            shots
        )

        support_indices.extend(selected)

    return support_indices


# --------------------------------------------------
# Extract normalized embeddings
# Must match evaluate_few_shot.py
# --------------------------------------------------

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
# Main
# --------------------------------------------------

def main():

    print("Device:", DEVICE)

    # Create output directory if necessary.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Use deterministic preprocessing.
    dataset = ImageFolder(
        TRAIN_DIR,
        transform=validation_transforms
    )

    print("Classes:", dataset.classes)

    if dataset.classes != EXPECTED_CLASSES:
        raise ValueError(
            "Unexpected class ordering!\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found:    {dataset.classes}"
        )

    support_indices = get_support_indices(
        dataset,
        SHOTS,
        SEED
    )

    print("Shots per class:", SHOTS)
    print("Total support images:", len(support_indices))
    print("Support seed:", SEED)

    # Load the existing pretrained ResNet18 feature extractor.
    feature_extractor = (
        create_feature_extractor()
        .to(DEVICE)
    )

    embeddings, labels = extract_embeddings(
        feature_extractor,
        dataset,
        support_indices
    )

    # Construct prototypes using the existing implementation.
    prototypes = create_prototypes(
        embeddings,
        labels,
        len(dataset.classes)
    )

    # Create a support-set signature for reproducibility.
    support_signature = hashlib.sha256(
        ",".join(
            map(str, sorted(support_indices))
        ).encode()
    ).hexdigest()

    # Save prototypes and their metadata.
    artifact = {
        "prototypes": prototypes.cpu(),
        "class_names": dataset.classes,
        "shots_per_class": SHOTS,
        "seed": SEED,
        "embedding_dimension": embeddings.shape[1],
        "distance_metric": "euclidean",
        "prototype_method": "mean_of_l2_normalized_embeddings",
        "feature_extractor": "ResNet18_ImageNet",
        "image_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "support_signature": support_signature,
    }

    torch.save(
        artifact,
        PROTOTYPE_PATH
    )

    # Save a human-readable manifest of selected support images.
    with SUPPORT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_index",
                "image_path",
                "class_name",
                "class_id",
                "seed",
            ]
        )

        writer.writeheader()

        for index in support_indices:

            image_path, class_id = dataset.samples[index]

            writer.writerow({
                "dataset_index": index,
                "image_path": str(image_path),
                "class_name": dataset.classes[class_id],
                "class_id": class_id,
                "seed": SEED,
            })

    # Display prototype information.
    print("\nPrototype generation completed.")

    print("Prototype tensor shape:", tuple(prototypes.shape))

    print("\nPrototype norms:")

    for class_id, class_name in enumerate(dataset.classes):

        norm = torch.linalg.vector_norm(
            prototypes[class_id]
        ).item()

        print(f"{class_name}: {norm:.6f}")

    print("\nSupport images per class:")

    for class_name in dataset.classes:

        count = sum(
            1
            for index in support_indices
            if dataset.samples[index][1]
            == dataset.class_to_idx[class_name]
        )

        print(f"{class_name}: {count}")

    print("\nSaved prototype artifact:")
    print(PROTOTYPE_PATH)

    print("\nSaved support manifest:")
    print(SUPPORT_CSV_PATH)

    print("\nSupport signature:")
    print(support_signature)


if __name__ == "__main__":
    main()