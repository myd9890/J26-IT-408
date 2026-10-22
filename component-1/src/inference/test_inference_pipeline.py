import csv
import json
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder

from src.data.transforms import validation_transforms
from src.inference.predictor import BananaDiseasePredictor


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TEST_DIR = Path("data/raw/test")

OUTPUT_DIR = Path("results/inference_pipeline_test")

BATCH_SIZE = 16


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def select_test_images(predictor, dataset):
    """
    Select one correctly classified image per class,
    where available, and one misclassified image,
    where available.

    Uses the existing predictor to make selections.
    """

    selected = []
    records = []

    print("\nEvaluating the test dataset...")

    for index, (image_path, true_label) in enumerate(
        dataset.samples
    ):

        result = predictor.predict(
            image_path,
            explain=False,
        )

        predicted_class = result["predicted_class"]
        true_class = dataset.classes[true_label]

        is_correct = predicted_class == true_class

        record = {
            "dataset_index": index,
            "image_path": str(image_path),
            "true_class": true_class,
            "predicted_class": predicted_class,
            "correct": is_correct,
            "nearest_distance_margin": (
                result["nearest_distance_margin"]
            ),
        }

        for class_name, distance in result["distances"].items():
            record[f"distance_{class_name}"] = distance

        records.append(record)

        if is_correct:
            if not any(
                item["true_class"] == true_class
                and item["correct"]
                for item in selected
            ):
                selected.append(record)

        else:
            if not any(
                not item["correct"]
                for item in selected
            ):
                selected.append(record)

    return selected, records


# --------------------------------------------------
# MAIN TEST
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictor = BananaDiseasePredictor()

    dataset = ImageFolder(
        TEST_DIR,
        transform=validation_transforms,
    )

    print("\nClasses:", dataset.classes)
    print("Total test images:", len(dataset))

    if dataset.classes != predictor.class_names:
        raise ValueError(
            "Dataset class ordering differs from predictor class ordering."
        )

    # --------------------------------------------------
    # Test 1: Evaluate all test images without Grad-CAM.
    # --------------------------------------------------

    selected_images, all_records = select_test_images(
        predictor,
        dataset,
    )

    accuracy = sum(
        record["correct"]
        for record in all_records
    ) / len(all_records)

    print("\n========== FULL TEST SET ==========")
    print(f"Images evaluated: {len(all_records)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save full test predictions.
    predictions_path = OUTPUT_DIR / "all_predictions.csv"

    fieldnames = list(all_records[0].keys())

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_records)

    # --------------------------------------------------
    # Test 2: Check prediction consistency.
    # --------------------------------------------------

    print("\n========== PREDICTION CONSISTENCY ==========")

    consistency_passed = True

    for record in selected_images:

        first_result = predictor.predict(
            record["image_path"],
            explain=False,
        )

        second_result = predictor.predict(
            record["image_path"],
            explain=False,
        )

        same_prediction = (
            first_result["predicted_class"]
            == second_result["predicted_class"]
        )

        same_distances = all(
            abs(
                first_result["distances"][class_name]
                - second_result["distances"][class_name]
            ) < 1e-6
            for class_name in predictor.class_names
        )

        passed = same_prediction and same_distances

        consistency_passed &= passed

        print(
            f"{Path(record['image_path']).name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # --------------------------------------------------
    # Test 3: Generate Grad-CAM for selected images.
    # --------------------------------------------------

    print("\n========== GRAD-CAM TEST ==========")

    explanation_results = []

    for record in selected_images:

        result = predictor.predict(
            record["image_path"],
            explain=True,
        )

        heatmap = result.get("gradcam")

        explanation_passed = (
            heatmap is not None
            and heatmap.shape == (224, 224)
            and torch.isfinite(
                torch.as_tensor(heatmap)
            ).all().item()
        )

        print(
            f"{Path(record['image_path']).name}: "
            f"{'PASS' if explanation_passed else 'FAIL'}"
        )

        explanation_results.append({
            "image_path": record["image_path"],
            "true_class": record["true_class"],
            "predicted_class": result["predicted_class"],
            "correct": record["correct"],
            "gradcam_generated": explanation_passed,
            "explanation_target": result.get(
                "explanation_target"
            ),
        })

    # --------------------------------------------------
    # Test 4: Confirm explain=False omits Grad-CAM.
    # --------------------------------------------------

    print("\n========== OPTIONAL EXPLANATION TEST ==========")

    sample_path = all_records[0]["image_path"]

    result_without_explanation = predictor.predict(
        sample_path,
        explain=False,
    )

    explanation_omitted = (
        "gradcam" not in result_without_explanation
        and "display_image" not in result_without_explanation
    )

    print(
        "Grad-CAM omitted when explain=False:",
        "PASS" if explanation_omitted else "FAIL",
    )

    # --------------------------------------------------
    # Save summary.
    # --------------------------------------------------

    summary = {
        "test_images": len(all_records),
        "correct_predictions": sum(
            record["correct"]
            for record in all_records
        ),
        "accuracy": accuracy,
        "prediction_consistency_passed": consistency_passed,
        "gradcam_tests": len(explanation_results),
        "gradcam_passed": sum(
            item["gradcam_generated"]
            for item in explanation_results
        ),
        "explain_false_passed": explanation_omitted,
        "selected_images": explanation_results,
    }

    summary_path = OUTPUT_DIR / "test_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n========== FINAL SUMMARY ==========")
    print(json.dumps(summary, indent=4))

    print("\nSaved predictions:", predictions_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()