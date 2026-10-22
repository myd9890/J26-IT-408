import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
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

SHOTS = 20
SEED = 42
BATCH_SIZE = 16
NUM_IMAGES_PER_CLASS = 5

OUTPUT_DIR = Path("results/gradcam_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# SUPPORT SET
# --------------------------------------------------

def get_support_indices(dataset, shots, seed):

    rng = random.Random(seed)
    support_indices = []

    for class_id in range(len(dataset.classes)):

        class_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]

        support_indices.extend(
            rng.sample(class_indices, shots)
        )

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

    all_embeddings = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            features = model(images)
            features = features.flatten(start_dim=1)

            embeddings = F.normalize(
                features,
                p=2,
                dim=1
            )

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    return (
        torch.cat(all_embeddings),
        torch.cat(all_labels)
    )


# --------------------------------------------------
# PROTOTYPE-BASED GRAD-CAM
# --------------------------------------------------

def generate_gradcam(model, image_tensor, prototypes, target_class):

    model.eval()

    activations = {}
    gradients = {}

    # Existing feature extractor:
    # layer4 is model[-2], before avgpool.
    target_layer = model[-2]

    def forward_hook(module, inputs, output):

        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.__setitem__("value", grad)
        )

    hook = target_layer.register_forward_hook(forward_hook)

    try:

        image = (
            image_tensor.unsqueeze(0)
            .to(DEVICE)
            .detach()
            .requires_grad_(True)
        )

        model.zero_grad(set_to_none=True)

        # Keep the forward pass differentiable.
        features = model(image)
        features = features.flatten(start_dim=1)

        embedding = F.normalize(
            features,
            p=2,
            dim=1
        )

        prototypes = prototypes.to(DEVICE)

        # Match the classifier's Euclidean-distance decision.
        distances = torch.cdist(
            embedding,
            prototypes,
            p=2
        )

        # The negative distance is the target score.
        target_score = -distances[0, target_class]

        model.zero_grad(set_to_none=True)
        target_score.backward()

        feature_maps = activations["value"]
        feature_gradients = gradients["value"]

        weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.relu(
            (weights * feature_maps).sum(
                dim=1,
                keepdim=True
            )
        )

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    finally:
        hook.remove()


# --------------------------------------------------
# IMAGE VISUALIZATION
# --------------------------------------------------

def save_visualization(
    image_index,
    test_dataset,
    image_tensor,
    true_label,
    predicted_label,
    distances,
    heatmap
):

    image_path = test_dataset.samples[image_index][0]

    # Use the original image, not the normalized tensor.
    original = Image.open(image_path).convert("RGB")
    original = original.resize((224, 224))

    original_array = np.asarray(original)

    class_names = test_dataset.classes

    correct = true_label == predicted_label

    status = "CORRECT" if correct else "MISCLASSIFIED"

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(original_array)
    axes[0].set_title(
        f"Original Image\nTrue: {class_names[true_label]}"
    )
    axes[0].axis("off")

    axes[1].imshow(original_array)
    axes[1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[1].set_title(
        f"Prototype Grad-CAM\nPredicted: {class_names[predicted_label]}"
    )
    axes[1].axis("off")

    distance_values = distances.detach().cpu().numpy()

    axes[2].bar(
        class_names,
        distance_values
    )
    axes[2].set_title("Euclidean Distance to Prototypes")
    axes[2].set_ylabel("Distance")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"{status} | Test index: {image_index}",
        fontsize=14
    )

    plt.tight_layout()

    filename = (
        f"image_{image_index:03d}_"
        f"{class_names[true_label]}_"
        f"{status.lower()}.png"
    )

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    train_dataset = ImageFolder(
        Path("data/processed/train"),
        transform=validation_transforms
    )

    _, _, test_dataset = create_datasets()

    print("Classes:", train_dataset.classes)
    print("Test images:", len(test_dataset))

    model = create_feature_extractor().to(DEVICE)
    model.eval()

    support_indices = get_support_indices(
        train_dataset,
        SHOTS,
        SEED
    )

    support_embeddings, support_labels = extract_embeddings(
        model,
        train_dataset,
        support_indices
    )

    prototypes = create_prototypes(
        support_embeddings,
        support_labels,
        len(train_dataset.classes)
    )

    # Evaluate all test images.
    test_embeddings, test_labels = extract_embeddings(
        model,
        test_dataset
    )

    distances = torch.cdist(
        test_embeddings,
        prototypes,
        p=2
    )

    predictions = distances.argmin(dim=1)

    accuracy = (
        predictions == test_labels
    ).float().mean().item()

    print(f"\n20-shot accuracy: {accuracy * 100:.2f}%")

    # Select up to five examples per true class.
    # This includes both correct and incorrect predictions.
    selected_indices = []

    for class_id in range(len(test_dataset.classes)):

        class_indices = [
            i for i, label in enumerate(test_labels)
            if label.item() == class_id
        ]

        # Prefer a mix of correct and incorrect predictions.
        correct_indices = [
            i for i in class_indices
            if predictions[i].item() == class_id
        ]

        incorrect_indices = [
            i for i in class_indices
            if predictions[i].item() != class_id
        ]

        selected = (
            incorrect_indices[:NUM_IMAGES_PER_CLASS]
            + correct_indices[:NUM_IMAGES_PER_CLASS]
        )

        # Limit total examples per true class.
        selected_indices.extend(
            selected[:NUM_IMAGES_PER_CLASS]
        )

    records = []

    print("\nGenerating explanations...")

    for image_index in selected_indices:

        image_tensor, true_label = test_dataset[image_index]

        predicted_label = predictions[image_index].item()

        heatmap = generate_gradcam(
            model,
            image_tensor,
            prototypes,
            predicted_label
        )

        output_path = save_visualization(
            image_index,
            test_dataset,
            image_tensor,
            true_label,
            predicted_label,
            distances[image_index],
            heatmap
        )

        records.append({
            "image_index": image_index,
            "image_path": test_dataset.samples[image_index][0],
            "true_class": test_dataset.classes[true_label],
            "predicted_class": test_dataset.classes[predicted_label],
            "correct": true_label == predicted_label,
            "true_class_distance": distances[
                image_index, true_label
            ].item(),
            "predicted_class_distance": distances[
                image_index, predicted_label
            ].item(),
            "explanation_path": str(output_path)
        })

        print(
            f"Image {image_index}: "
            f"True={test_dataset.classes[true_label]}, "
            f"Predicted={test_dataset.classes[predicted_label]}, "
            f"Saved={output_path}"
        )

    results_path = OUTPUT_DIR / "explanation_evaluation.csv"

    pd.DataFrame(records).to_csv(
        results_path,
        index=False
    )

    print("\nExplanation evaluation completed.")
    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()