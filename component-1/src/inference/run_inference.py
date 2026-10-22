import argparse
import json
from pathlib import Path

from src.inference.predictor import BananaDiseasePredictor
from src.inference.result_saver import save_inference_result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Screen banana leaf images using CLIP and predict "
            "disease using the ResNet18 prototypical classifier."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the image to screen.",
    )

    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="Skip Grad-CAM generation.",
    )

    parser.add_argument(
        "--output",
        default="results/inference",
        help="Directory where results will be saved.",
    )

    parser.add_argument(
        "--disable-gatekeeper",
        action="store_true",
        help="Skip CLIP screening and run disease classification directly.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    predictor = BananaDiseasePredictor(
        enable_gatekeeper=not args.disable_gatekeeper
    )

    result = predictor.predict(
        image_path,
        explain=not args.no_explanation,
    )

    print("\n========== INFERENCE RESULT ==========")
    print("Image:", result["image_path"])

    # Display CLIP screening information when available.
    if "gatekeeper" in result:

        gatekeeper = result["gatekeeper"]

        print("\n========== CLIP GATEKEEPER ==========")
        print("Screening status:", gatekeeper["status"])
        print(
            "Banana leaf score:",
            f"{gatekeeper['banana_leaf_score']:.6f}",
        )
        print(
            "Strongest negative score:",
            f"{gatekeeper['strongest_negative_score']:.6f}",
        )
        print(
            "Best matching description:",
            gatekeeper["best_matching_description"],
        )

    # Handle rejected or uncertain images safely.
    if result.get("predicted_class") is None:

        print("\nDisease classification was not performed.")
        print("Status:", result["status"])
        print("Message:", result.get("message", ""))

    else:

        print("\n========== DISEASE PREDICTION ==========")
        print("Predicted class:", result["predicted_class"])

        print(
            "Model confidence:",
            f"{result['model_confidence_percentage']:.2f}%",
        )

        print("\nClass scores:")

        for class_name, score in result["class_scores"].items():
            print(f"  {class_name}: {score:.6f}")

        print("\nPrototype distances:")

        for class_name, distance in result["distances"].items():
            print(f"  {class_name}: {distance:.6f}")

        print(
            "\nNearest-distance margin:",
            f"{result['nearest_distance_margin']:.6f}",
        )

        print("Status:", result["status"])

        if "explanation_target" in result:
            print(
                "Explanation target:",
                result["explanation_target"],
            )

    # Print the complete result as JSON for debugging.
    # NumPy arrays, such as Grad-CAM heatmaps, are excluded here.
    printable_result = {
        key: value
        for key, value in result.items()
        if key not in ("gradcam", "display_image")
    }

    print("\n========== RESULT JSON ==========")
    print(json.dumps(printable_result, indent=4, default=str))

    # Save the result.
    save_inference_result(
        result,
        output_dir=args.output,
    )

    print("\nSaved inference result to:", args.output)


if __name__ == "__main__":
    main()

