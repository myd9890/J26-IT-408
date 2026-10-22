import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
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

OUTPUT_DIR = Path(
    "results/gradcam_plus_plus_prototype"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLASS_NAMES = [
    "black_sigatoka",
    "healthy",
    "yellow_sigatoka",
]

# Five Black Sigatoka images previously misclassified
# by the ResNet18 20-shot, seed-42 classifier.
TARGET_IMAGES = [
    "00011.jpg",
    "00012.jpg",
    "00013.jpg",
    "00015.jpg",
    "00016.jpg",
]


# ============================================================
# SUPPORT SET SELECTION
# Must match evaluate_few_shot.py
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

            labels.append(batch_labels)

    return (
        torch.cat(embeddings),
        torch.cat(labels)
    )


# ============================================================
# GRAD-CAM++ IMPLEMENTATION
# ============================================================

class GradCAMPlusPlus:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None

        self.hook = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

        # This is conditional because ordinary prediction passes may run
        # under torch.no_grad(). The explanation pass enables gradients.
        if self.activations.requires_grad:
            self.activations.retain_grad()

    def generate(
        self,
        image_tensor,
        prototype,
        target_class
    ):
        # Explicitly enable autograd even if the caller is inside
        # torch.no_grad(). The feature-extractor weights remain frozen.
        with torch.enable_grad():

            self.model.eval()
            self.model.zero_grad(set_to_none=True)

            # Reset captured activations for this forward pass.
            self.activations = None

            # Since all model parameters are frozen, the input must
            # require gradients for autograd to track the forward pass.
            image_tensor = (
                image_tensor
                .unsqueeze(0)
                .to(DEVICE)
                .detach()
                .requires_grad_(True)
            )

            embedding = self.model(image_tensor)
            embedding = embedding.flatten(start_dim=1)

            embedding = F.normalize(
                embedding,
                p=2,
                dim=1
            )

            prototype = prototype.detach().to(DEVICE)

            # Larger scores indicate a closer distance to the target prototype.
            target_score = -torch.norm(
                embedding - prototype.unsqueeze(0),
                p=2,
                dim=1
            )[0]

            if not target_score.requires_grad:
                raise RuntimeError(
                    "Target score has no gradient. Check gradient tracking."
                )

            if self.activations is None:
                raise RuntimeError(
                    "Target layer activations were not captured."
                )

            if not self.activations.requires_grad:
                raise RuntimeError(
                    "Target layer activations do not require gradients."
                )

            # Obtain the score gradient with respect to the target-layer
            # activations directly. This avoids relying on .grad being
            # populated on a non-leaf tensor.
            gradients = torch.autograd.grad(
                outputs=target_score,
                inputs=self.activations,
                retain_graph=False,
                create_graph=False,
                allow_unused=False
            )[0]

            activations = self.activations.detach()[0]
            gradients = gradients.detach()[0]

            # Activations and gradients have shape [channels, height, width].
            gradients_squared = gradients.pow(2)
            gradients_cubed = gradients.pow(3)

            denominator = (
                2.0 * gradients_squared
                + (
                    activations * gradients_cubed
                ).sum(
                    dim=(1, 2),
                    keepdim=True
                )
            )

            eps = 1e-8

            alpha = gradients_squared / (
                denominator + eps
            )

            positive_gradients = F.relu(gradients)

            weights = (
                alpha * positive_gradients
            ).sum(dim=(1, 2))

            cam = (
                weights[:, None, None] * activations
            ).sum(dim=0)

            cam = F.relu(cam)
            cam = cam.cpu().numpy()

            # Normalize the heatmap safely.
            cam_min = float(cam.min())
            cam_max = float(cam.max())

            if np.isfinite(cam_min) and np.isfinite(cam_max) and (
                cam_max - cam_min > eps
            ):
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam, dtype=np.float32)

            if not np.isfinite(cam).all():
                raise RuntimeError(
                    "Grad-CAM++ generated a heatmap containing NaN or infinity."
                )

            return cam, float(target_score.detach().cpu())

    def close(self):

        self.hook.remove()


# ============================================================
# IMAGE VISUALIZATION
# ============================================================

def save_visualization(
    image_path,
    heatmap,
    actual_class,
    predicted_class,
    target_class,
    target_score,
    output_path
):

    original = Image.open(
        image_path
    ).convert("RGB")

    original = original.resize(
        (224, 224)
    )

    original_array = np.asarray(
        original
    )

    heatmap_image = Image.fromarray(
        np.uint8(
            np.clip(heatmap, 0, 1) * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    heatmap_array = (
        np.asarray(heatmap_image)
        .astype(np.float32)
        / 255.0
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        original_array
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[1].imshow(
        original_array
    )

    axes[1].imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        f"Grad-CAM++: {target_class}"
    )

    axes[2].imshow(
        heatmap_array,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Grad-CAM++ Heatmap"
    )

    for axis in axes:

        axis.axis("off")

    fig.suptitle(
        f"Actual: {actual_class} | "
        f"Predicted: {predicted_class}\n"
        f"Explanation target: {target_class} | "
        f"Target score: {target_score:.4f}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Using device:", DEVICE)

    # Use the same dataset setup as existing experiments.
    _, _, test_dataset = create_datasets()

    train_dataset = ImageFolder(
        TRAIN_PATH,
        transform=validation_transforms
    )

    if train_dataset.classes != CLASS_NAMES:

        raise ValueError(
            f"Unexpected class ordering: "
            f"{train_dataset.classes}"
        )

    # --------------------------------------------------------
    # Build the 20-shot, seed-42 prototypes.
    # --------------------------------------------------------

    support_indices = get_support_indices(
        train_dataset,
        SHOTS,
        SEED
    )

    print("\nSupport images:", len(support_indices))

    model = create_feature_extractor().to(DEVICE)

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
    ).to(DEVICE)

    # --------------------------------------------------------
    # Find selected test images.
    # --------------------------------------------------------

    selected_indices = []

    for index, (image_path, _) in enumerate(
        test_dataset.samples
    ):

        if Path(image_path).name in TARGET_IMAGES:

            selected_indices.append(index)

    found_names = {
        Path(test_dataset.samples[index][0]).name
        for index in selected_indices
    }

    missing_names = (
        set(TARGET_IMAGES) - found_names
    )

    if missing_names:

        raise FileNotFoundError(
            f"Test images not found: {sorted(missing_names)}"
        )

    # --------------------------------------------------------
    # Prepare Grad-CAM++.
    # --------------------------------------------------------

    # For the ResNet18 sequential feature extractor,
    # model[-2] is layer4.
    target_layer = model[-2]

    gradcam = GradCAMPlusPlus(
        model,
        target_layer
    )

    records = []

    try:

        for index in selected_indices:

            image_path, actual_id = (
                test_dataset.samples[index]
            )

            image_tensor, _ = test_dataset[index]

            image_batch = (
                image_tensor
                .unsqueeze(0)
                .to(DEVICE)
            )

            # Determine the original model prediction.
            with torch.no_grad():

                embedding = model(
                    image_batch
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

            actual_class = CLASS_NAMES[
                actual_id
            ]

            predicted_class = CLASS_NAMES[
                predicted_id
            ]

            print("\n" + "=" * 70)

            print(
                "Image:",
                Path(image_path).name
            )

            print(
                "Actual:",
                actual_class
            )

            print(
                "Predicted:",
                predicted_class
            )

            print(
                "Distances:",
                {
                    CLASS_NAMES[class_id]: round(
                        distances[class_id].item(),
                        4
                    )
                    for class_id in range(
                        len(CLASS_NAMES)
                    )
                }
            )

            # Generate two explanations:
            # 1. The predicted class.
            # 2. The actual class.
            targets = {
                "predicted": predicted_id,
                "actual": actual_id
            }

            for explanation_type, target_id in (
                targets.items()
            ):

                target_class = CLASS_NAMES[
                    target_id
                ]

                heatmap, target_score = (
                    gradcam.generate(
                        image_tensor,
                        prototypes[target_id],
                        target_class
                    )
                )

                filename = (
                    f"{Path(image_path).stem}_"
                    f"{explanation_type}_"
                    f"{target_class}_gradcampp.png"
                )

                output_path = (
                    OUTPUT_DIR / filename
                )

                save_visualization(
                    image_path=image_path,
                    heatmap=heatmap,
                    actual_class=actual_class,
                    predicted_class=predicted_class,
                    target_class=target_class,
                    target_score=target_score,
                    output_path=output_path
                )

                records.append({
                    "image": Path(image_path).name,
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "correct": actual_id == predicted_id,
                    "explanation_type": explanation_type,
                    "target_class": target_class,
                    "target_score": target_score,
                    "distance_black_sigatoka": distances[0].item(),
                    "distance_healthy": distances[1].item(),
                    "distance_yellow_sigatoka": distances[2].item(),
                    "heatmap_file": filename
                })

                print(
                    f"Saved explanation: {filename}"
                )

    finally:

        gradcam.close()

    # --------------------------------------------------------
    # Save metadata.
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR / "gradcampp_results.csv"
    )

    with csv_path.open(
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

    print("\n" + "=" * 70)

    print(
        "Grad-CAM++ analysis completed."
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print(
        "Results CSV:",
        csv_path
    )

    print(
        "Images processed:",
        len(selected_indices)
    )

    print(
        "Explanations generated:",
        len(records)
    )


if __name__ == "__main__":

    main()