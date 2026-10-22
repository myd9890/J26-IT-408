import torch

from src.models.dinov2_feature_extractor import (
    create_dinov2_feature_extractor
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = create_dinov2_feature_extractor(device)

    # Simulate a batch of four RGB images.
    dummy_images = torch.randn(4, 3, 224, 224).to(device)

    with torch.no_grad():
        embeddings = model(dummy_images)

    print("\n========== DINOv2 TEST ==========")
    print("Input shape:", dummy_images.shape)
    print("Embedding shape:", embeddings.shape)
    print("All embeddings finite:", torch.isfinite(embeddings).all().item())
    print("=================================")


if __name__ == "__main__":
    main()

