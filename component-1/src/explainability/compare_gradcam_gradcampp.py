import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SHOTS = 20
BATCH_SIZE = 16

TRAIN_PATH = Path("data/processed/train")
OUTPUT_DIR = Path("results/gradcam_vs_gradcampp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["black_sigatoka", "healthy", "yellow_sigatoka"]

# The five Black Sigatoka images misclassified as Healthy in
# the existing ResNet18 20-shot, seed-42 experiment.
TARGET_IMAGES = ["00011.jpg", "00012.jpg", "00013.jpg", "00015.jpg", "00016.jpg"]


# ============================================================
# REPRODUCIBLE SUPPORT SET (same logic as evaluate_few_shot.py)
# ============================================================

def get_support_indices(dataset, shots, seed):
    random.seed(seed)
    support_indices = []

    for class_id in range(len(dataset.classes)):
        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]
        selected = random.sample(class_indices, shots)
        support_indices.extend(selected)

    return support_indices


def extract_embeddings(model, dataset, indices, batch_size=16):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    embeddings = []
    labels = []
    model.eval()

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(DEVICE)
            features = model(images).flatten(start_dim=1)
            features = F.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu())
            labels.append(batch_labels)

    return torch.cat(embeddings), torch.cat(labels)


# ============================================================
# SAME SCORE + SAME ACTIVATIONS FOR BOTH EXPLANATION METHODS
# ============================================================

class ActivationCapture:
    def __init__(self, layer):
        self.activation = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activation = output

    def close(self):
        self.handle.remove()


def normalize_cam(cam, eps=1e-8):
    cam = np.asarray(cam, dtype=np.float32)
    if not np.isfinite(cam).all():
        raise RuntimeError("CAM contains NaN or infinity.")

    cam = np.maximum(cam, 0)
    low = float(cam.min())
    high = float(cam.max())

    if high - low <= eps:
        return np.zeros_like(cam, dtype=np.float32)

    return (cam - low) / (high - low)


def generate_both_cams(model, target_layer, image_tensor, prototype):
    """
    Calculate Grad-CAM and an experimental Grad-CAM++ approximation
    from the same forward pass, target layer, prototype, and target score.

    Note: Grad-CAM++ here uses the same gradient-power approximation as
    the previous experimental script. It is not a guarantee of exact
    equivalence to every standard Grad-CAM++ implementation.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    capture = ActivationCapture(target_layer)

    try:
        with torch.enable_grad():
            image = (
                image_tensor.unsqueeze(0)
                .to(DEVICE)
                .detach()
                .requires_grad_(True)
            )

            embedding = model(image).flatten(start_dim=1)
            embedding = F.normalize(embedding, p=2, dim=1)

            target_score = -torch.norm(
                embedding - prototype.detach().to(DEVICE).unsqueeze(0),
                p=2,
                dim=1,
            )[0]

            activations = capture.activation

            if activations is None or not activations.requires_grad:
                raise RuntimeError(
                    "Target-layer activations were not captured with gradients."
                )

            gradients = torch.autograd.grad(
                outputs=target_score,
                inputs=activations,
                retain_graph=False,
                create_graph=False,
            )[0]

            # Shape: [channels, height, width]
            A = activations.detach()[0]
            G = gradients.detach()[0]

            # -------------------------
            # Standard Grad-CAM
            # -------------------------
            gradcam_weights = G.mean(dim=(1, 2))
            gradcam_map = (gradcam_weights[:, None, None] * A).sum(dim=0)
            gradcam_map = F.relu(gradcam_map).cpu().numpy()
            gradcam_map = normalize_cam(gradcam_map)

            # -------------------------
            # Grad-CAM++ approximation
            # -------------------------
            G2 = G.pow(2)
            G3 = G.pow(3)

            denominator = (
                2.0 * G2
                + (A * G3).sum(dim=(1, 2), keepdim=True)
            )

            eps = 1e-8
            alpha = G2 / (denominator + eps)
            positive_gradients = F.relu(G)
            weights_pp = (alpha * positive_gradients).sum(dim=(1, 2))

            gradcampp_map = (
                weights_pp[:, None, None] * A
            ).sum(dim=0)

            gradcampp_map = F.relu(gradcampp_map).cpu().numpy()
            gradcampp_map = normalize_cam(gradcampp_map)

            score_value = float(target_score.detach().cpu())

            return gradcam_map, gradcampp_map, score_value

    finally:
        capture.close()


def resize_heatmap(heatmap, size=(224, 224)):
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    )
    heatmap_image = heatmap_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(heatmap_image, dtype=np.float32) / 255.0


def save_comparison(
    image_path,
    gradcam_map,
    gradcampp_map,
    actual_class,
    predicted_class,
    target_class,
    target_score,
    output_path,
):
    original = Image.open(image_path).convert("RGB").resize((224, 224))
    original_array = np.asarray(original)

    gradcam_heatmap = resize_heatmap(gradcam_map)
    gradcampp_heatmap = resize_heatmap(gradcampp_map)

    fig, axes = plt.subplots(1, 5, figsize=(23, 5))

    axes[0].imshow(original_array)
    axes[0].set_title("Original")

    axes[1].imshow(original_array)
    axes[1].imshow(
        gradcam_heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1
    )
    axes[1].set_title("Grad-CAM overlay")

    axes[2].imshow(original_array)
    axes[2].imshow(
        gradcampp_heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1
    )
    axes[2].set_title("Grad-CAM++ overlay")

    axes[3].imshow(gradcam_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Grad-CAM map")

    axes[4].imshow(gradcampp_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[4].set_title("Grad-CAM++ map")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{Path(image_path).name} | Actual: {actual_class} | "
        f"Predicted: {predicted_class}\n"
        f"Shared explanation target: {target_class} | "
        f"Shared target score: {target_score:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Using device:", DEVICE)

    _, _, test_dataset = create_datasets()
    train_dataset = ImageFolder(
        TRAIN_PATH, transform=validation_transforms
    )

    if train_dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {train_dataset.classes}")

    support_indices = get_support_indices(train_dataset, SHOTS, SEED)
    print(f"Support images: {len(support_indices)} ({SHOTS}-shot, seed {SEED})")

    model = create_feature_extractor().to(DEVICE)
    model.eval()

    support_embeddings, support_labels = extract_embeddings(
        model, train_dataset, support_indices, BATCH_SIZE
    )

    prototypes = create_prototypes(
        support_embeddings, support_labels, len(CLASS_NAMES)
    ).to(DEVICE)

    # In the existing ResNet18 sequential feature extractor, model[-2]
    # is layer4, matching the previous Grad-CAM/Grad-CAM++ experiments.
    target_layer = model[-2]

    indices_by_name = {
        Path(path).name: index
        for index, (path, _) in enumerate(test_dataset.samples)
    }

    missing = sorted(set(TARGET_IMAGES) - set(indices_by_name))
    if missing:
        raise FileNotFoundError(f"Test images not found: {missing}")

    csv_records = []

    for filename in TARGET_IMAGES:
        index = indices_by_name[filename]
        image_path, actual_id = test_dataset.samples[index]
        image_tensor, _ = test_dataset[index]

        # Calculate prediction and distances using the same model/prototypes.
        with torch.no_grad():
            embedding = model(image_tensor.unsqueeze(0).to(DEVICE))
            embedding = F.normalize(embedding.flatten(start_dim=1), p=2, dim=1)
            distances = torch.cdist(embedding.cpu(), prototypes.cpu(), p=2)[0]
            predicted_id = int(distances.argmin().item())

        actual_class = CLASS_NAMES[actual_id]
        predicted_class = CLASS_NAMES[predicted_id]

        # For a fair direct comparison, both methods explain the same
        # predicted-class score. For these known errors, that is "healthy".
        target_id = predicted_id
        target_class = CLASS_NAMES[target_id]

        gradcam_map, gradcampp_map, target_score = generate_both_cams(
            model,
            target_layer,
            image_tensor,
            prototypes[target_id],
        )

        output_path = OUTPUT_DIR / f"{Path(filename).stem}_comparison.png"

        save_comparison(
            image_path=image_path,
            gradcam_map=gradcam_map,
            gradcampp_map=gradcampp_map,
            actual_class=actual_class,
            predicted_class=predicted_class,
            target_class=target_class,
            target_score=target_score,
            output_path=output_path,
        )

        csv_records.append({
            "image": filename,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "explanation_target": target_class,
            "target_score": target_score,
            "distance_black_sigatoka": float(distances[0]),
            "distance_healthy": float(distances[1]),
            "distance_yellow_sigatoka": float(distances[2]),
            "gradcam_file": output_path.name,
            "gradcam_heatmap_min": float(gradcam_map.min()),
            "gradcam_heatmap_max": float(gradcam_map.max()),
            "gradcampp_heatmap_min": float(gradcampp_map.min()),
            "gradcampp_heatmap_max": float(gradcampp_map.max()),
        })

        print(
            f"{filename}: actual={actual_class}, predicted={predicted_class}, "
            f"target={target_class}, score={target_score:.4f}"
        )
        print(f"  Saved: {output_path}")

    csv_path = OUTPUT_DIR / "comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_records[0].keys())
        writer.writeheader()
        writer.writerows(csv_records)

    print("\nComparison complete.")
    print("Output directory:", OUTPUT_DIR)
    print("Summary CSV:", csv_path)
    print("Generated comparisons:", len(csv_records))


if __name__ == "__main__":
    main()