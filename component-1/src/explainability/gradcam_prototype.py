import random
from pathlib import Path

import numpy as np
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

OUTPUT_DIR = Path("results/gradcam_prototype")

CLASS_NAMES = [
    "black_sigatoka",
    "healthy",
    "yellow_sigatoka",
]


# --------------------------------------------------
# SUPPORT SET SELECTION
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

        selected = random.sample(
            class_indices,
            shots
        )

        support_indices.extend(selected)

    return support_indices


# --------------------------------------------------
# EMBEDDING EXTRACTION
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

            features = features.flatten(start_dim=1)

            features = F.normalize(
                features,
                p=2,
                dim=1
            )

            embeddings.append(features.cpu())
            labels.append(batch_labels)

    return (
        torch.cat(embeddings, dim=0),
        torch.cat(labels, dim=0)
    )


# --------------------------------------------------
# GRAD-CAM FOR PROTOTYPE CLASSIFICATION
# --------------------------------------------------

def generate_gradcam(model, image_tensor, prototypes, target_class):

    model.eval()

    activations = {}
    gradients = {}

    # In the existing ResNet18 feature extractor:
    # [-1] is avgpool and [-2] is layer4.
    target_layer = model[-2]

    def forward_hook(module, inputs, output):

        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.__setitem__(
                "value", grad
            )
        )

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        # Enable input gradients even though model weights
        # are frozen.
        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
            .detach()
            .requires_grad_(True)
        )

        model.zero_grad(set_to_none=True)

        # Forward pass through the existing feature extractor.
        features = model(image_tensor)

        features = features.flatten(start_dim=1)

        embeddings = F.normalize(
            features,
            p=2,
            dim=1
        )

        prototypes = prototypes.to(DEVICE)

        # Use the same Euclidean-distance concept as the
        # existing prototypical classifier.
        distances = torch.cdist(
            embeddings,
            prototypes
        )

        predicted_class = distances.argmin(dim=1).item()

        # Explain the predicted class by differentiating
        # its negative distance.
        score = -distances[0, target_class]

        score.backward()

        feature_maps = activations["value"]
        feature_gradients = gradients["value"]

        # Average gradients spatially to obtain channel weights.
        weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of convolutional feature maps.
        cam = torch.relu(
            (weights * feature_maps).sum(
                dim=1,
                keepdim=True
            )
        )

        # Upsample the heatmap to the input image dimensions.
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize the heatmap to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class

    finally:
        hook.remove()


# --------------------------------------------------
# SELECT TEST IMAGES
# --------------------------------------------------

def select_test_images(
    predictions,
    test_labels,
    test_dataset
):

    selected_indices = []

    for class_id in range(len(test_dataset.classes)):

        # Prefer a correctly classified example.
        correct_indices = [
            i
            for i in range(len(test_dataset))
            if test_labels[i].item() == class_id
            and predictions[i].item() == class_id
        ]

        if correct_indices:
            selected_indices.append(correct_indices[0])

        else:
            # If no correct example exists, use the first
            # example belonging to that true class.
            class_indices = [
                i
                for i in range(len(test_dataset))
                if test_labels[i].item() == class_id
            ]

            if class_indices:
                selected_indices.append(class_indices[0])

    return selected_indices


# --------------------------------------------------
# VISUALIZE AND SAVE EXPLANATIONS
# --------------------------------------------------

def save_explanation(
    model,
    image_index,
    test_dataset,
    prototypes
):

    image_tensor, true_label = test_dataset[image_index]

    image_path = test_dataset.samples[image_index][0]

    # Generate Grad-CAM for the model's predicted class.
    with torch.no_grad():

        image_batch = image_tensor.unsqueeze(0).to(DEVICE)

        features = model(image_batch)

        features = features.flatten(start_dim=1)

        embeddings = F.normalize(
            features,
            p=2,
            dim=1
        )

        distances = torch.cdist(
            embeddings,
            prototypes.to(DEVICE)
        )

        predicted_class = distances.argmin(dim=1).item()

    cam, gradcam_class = generate_gradcam(
        model,
        image_tensor,
        prototypes,
        predicted_class
    )

    # Load the original image for display.
    original_image = Image.open(
        image_path
    ).convert("RGB").resize((224, 224))

    original_array = np.asarray(original_image)

    true_name = test_dataset.classes[true_label]
    predicted_name = test_dataset.classes[predicted_class]

    # Produce a side-by-side comparison.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    axes[0].imshow(original_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(original_array)
    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        f"Prototype Grad-CAM\n"
        f"True: {true_name}\n"
        f"Predicted: {predicted_name}"
    )

    axes[1].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{CLASS_NAMES[true_label]}_example.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nImage: {image_path}")
    print(f"True class: {true_name}")
    print(f"Predicted class: {predicted_name}")
    print(f"Grad-CAM target class: {test_dataset.classes[gradcam_class]}")
    print(f"Saved explanation: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load the training data with deterministic transforms.
    train_dataset = ImageFolder(
        Path("data/processed/train"),
        transform=validation_transforms
    )

    # Load the untouched test dataset.
    _, _, test_dataset = create_datasets()

    print("Classes:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    # Load the same pretrained ResNet18 feature extractor
    # used in the few-shot experiments.
    model = create_feature_extractor().to(DEVICE)
    model.eval()

    # Select the 20-shot support set with seed 42.
    support_indices = get_support_indices(
        train_dataset,
        SHOTS,
        SEED
    )

    print(
        f"\nBuilding prototypes using "
        f"{SHOTS}-shot support examples..."
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

    # Extract test embeddings and obtain predictions.
    test_embeddings, test_labels = extract_embeddings(
        model,
        test_dataset
    )

    distances = torch.cdist(
        test_embeddings,
        prototypes
    )

    predictions = distances.argmin(dim=1)

    accuracy = (
        predictions == test_labels
    ).float().mean().item()

    print(
        f"\n20-shot test accuracy for this support set: "
        f"{accuracy * 100:.2f}%"
    )

    # Choose one representative test image per true class.
    selected_indices = select_test_images(
        predictions,
        test_labels,
        test_dataset
    )

    print("\nGenerating prototype-based Grad-CAM...")

    for image_index in selected_indices:

        save_explanation(
            model,
            image_index,
            test_dataset,
            prototypes
        )

    print("\nGrad-CAM generation completed.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()