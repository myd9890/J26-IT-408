import matplotlib.pyplot as plt

from src.inference.predictor import BananaDiseasePredictor


def main():

    predictor = BananaDiseasePredictor()

    image_path = (
        "data/raw/test/black_sigatoka/00011.jpg"
    )

    result = predictor.predict(
        image_path,
        explain=True,
    )

    print("\n========== INFERENCE RESULT ==========")

    for key, value in result.items():

        if key not in {"gradcam", "display_image"}:
            print(f"{key}: {value}")

    # Display the original image and Grad-CAM overlay.
    original = result["display_image"]
    heatmap = result["gradcam"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(original)
    axes[1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        f"Prototype Grad-CAM\n"
        f"Target: {result['explanation_target']}"
    )

    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()