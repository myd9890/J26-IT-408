import numpy as np
import json

from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from src.data.transforms import validation_transforms
from src.models.feature_extractor import create_feature_extractor
from src.models.prototypical import classify_with_prototypes
from src.inference.image_validator import validate_image
from src.inference.clip_gatekeeper import BananaLeafGatekeeper
from src.explainability.gradcam_inference import generate_gradcam


class BananaDiseasePredictor:

    def __init__(
        self,
        prototype_path="models/resnet18_prototypes_20shot_seed42.pth",
        calibration_path="models/resnet18_confidence_calibration.json",
        review_margin_threshold=None,
        review_confidence_threshold=None,
        enable_gatekeeper=True,
        gatekeeper_threshold=0.30,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.prototype_path = Path(prototype_path)

        if not self.prototype_path.exists():
            raise FileNotFoundError(
                f"Prototype file not found: {self.prototype_path}"
            )

        self.review_margin_threshold = review_margin_threshold
        self.review_confidence_threshold = review_confidence_threshold

        # Load saved prototypes.
        artifact = torch.load(
            self.prototype_path,
            map_location="cpu",
            weights_only=False,
        )

        self.prototypes = artifact["prototypes"].to(self.device)
        self.class_names = artifact["class_names"]

        if self.prototypes.shape != (
            len(self.class_names),
            512,
        ):
            raise ValueError(
                "Unexpected prototype tensor shape: "
                f"{tuple(self.prototypes.shape)}"
            )

        # Load confidence calibration.
        self.calibration_path = Path(calibration_path)

        if not self.calibration_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {self.calibration_path}"
            )

        with open(self.calibration_path, "r", encoding="utf-8") as file:
            calibration = json.load(file)

        self.temperature = float(calibration["temperature"])

        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Calibration temperature must be positive.")

        # Load ResNet18 feature extractor.
        self.feature_extractor = (
            create_feature_extractor()
            .to(self.device)
        )

        self.feature_extractor.eval()
        self.transform = validation_transforms

        # Load CLIP once during initialization.
        self.enable_gatekeeper = enable_gatekeeper
        self.gatekeeper = None

        if self.enable_gatekeeper:
            self.gatekeeper = BananaLeafGatekeeper(
                threshold=gatekeeper_threshold
            )

        print("Banana disease predictor initialized.")
        print("Device:", self.device)
        print("Classes:", self.class_names)
        print("CLIP gatekeeper enabled:", self.enable_gatekeeper)

    def predict(
        self,
        image_path,
        explain=True,
    ):

        image_path = Path(image_path)

        # Stage 1: Validate image file.
        validate_image(image_path)

        try:
            with Image.open(image_path) as image:
                original_image = image.convert("RGB")

        except Exception as exc:
            raise ValueError(
                f"Could not process image: {image_path}"
            ) from exc

        # Stage 2: CLIP banana-leaf screening.
        if self.enable_gatekeeper:

            gatekeeper_result = self.gatekeeper.predict(
                image_path
            )

            if (
                gatekeeper_result["status"]
                != "accepted_for_disease_screening"
            ):

                gatekeeper_status = gatekeeper_result["status"]

                if gatekeeper_status == "uncertain":
                    result_status = "gatekeeper_uncertain"
                    message = (
                        "The image could not be confidently verified "
                        "as a banana leaf. Please upload a clearer "
                        "image showing the banana leaf."
                    )
                else:
                    result_status = "gatekeeper_rejected"
                    message = (
                        "This image was not accepted for banana disease "
                        "screening. Please upload an image of a banana leaf."
                    )

                return {
                    "image_path": str(image_path),
                    "predicted_class": None,
                    "status": result_status,
                    "message": message,
                    "gatekeeper": gatekeeper_result,
                }

        else:
            gatekeeper_result = None

        # Stage 3: Preprocess for ResNet18.
        try:
            image_tensor = self.transform(original_image)
            display_image = original_image.resize((224, 224))

        except Exception as exc:
            raise ValueError(
                f"Could not preprocess image: {image_path}"
            ) from exc

        image_tensor = image_tensor.to(self.device)

        # Stage 4: Few-shot disease classification.
        with torch.no_grad():

            features = self.feature_extractor(
                image_tensor.unsqueeze(0)
            )

            features = features.flatten(start_dim=1)

            embeddings = F.normalize(
                features,
                p=2,
                dim=1,
            )

            predictions, distances = classify_with_prototypes(
                embeddings,
                self.prototypes,
            )

            confidence_scores = torch.softmax(
                -distances[0] / self.temperature,
                dim=0,
            )

        predicted_index = predictions.item()
        distance_values = distances[0].cpu().tolist()

        sorted_distances = sorted(distance_values)
        margin = sorted_distances[1] - sorted_distances[0]

        predicted_confidence = float(
            confidence_scores[predicted_index].item()
        )

        # Stage 5: Optional confidence and margin review.
        status = "predicted"

        if (
            self.review_margin_threshold is not None
            and margin < self.review_margin_threshold
        ):
            status = "review_recommended"

        if (
            self.review_confidence_threshold is not None
            and predicted_confidence < self.review_confidence_threshold
        ):
            status = "review_recommended"

        # Stage 6: Build prediction result.
        result = {
            "image_path": str(image_path),
            "predicted_class": self.class_names[predicted_index],

            # Model score, not a guaranteed probability of correctness.
            "model_confidence_score": predicted_confidence,
            "model_confidence_percentage": round(
                predicted_confidence * 100,
                2,
            ),

            "class_scores": {
                class_name: float(score)
                for class_name, score in zip(
                    self.class_names,
                    confidence_scores.cpu().tolist(),
                )
            },

            "distances": {
                class_name: float(distance)
                for class_name, distance in zip(
                    self.class_names,
                    distance_values,
                )
            },

            "nearest_distance_margin": float(margin),
            "status": status,
        }

        if gatekeeper_result is not None:
            result["gatekeeper"] = gatekeeper_result

        # Stage 7: Generate Grad-CAM for classified images only.
        if explain:

            heatmap = generate_gradcam(
                model=self.feature_extractor,
                image_tensor=image_tensor,
                prototypes=self.prototypes,
                target_class=predicted_index,
                device=self.device,
            )

            result["gradcam"] = heatmap

            result["explanation_target"] = (
                self.class_names[predicted_index]
            )

            result["display_image"] = np.asarray(
                display_image
            )

        return result

