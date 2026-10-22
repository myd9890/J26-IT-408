from pathlib import Path

import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
)

from src.inference.image_validator import validate_image


class BananaLeafGatekeeper:

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        threshold=0.30,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.threshold = threshold

        self.model = CLIPModel.from_pretrained(
            model_name,
            local_files_only=True,
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )

        self.model.eval()

        # Candidate descriptions for zero-shot comparison.
        self.positive_prompts = [
            "a photo of a banana leaf",
            "a close-up photo of a Cavendish banana leaf",
            "a photo of a large green banana leaf with a central midrib",
            "a photo of a banana leaf with parallel veins",
            "a photo of a banana leaf affected by black Sigatoka disease",
            "a photo of a banana leaf with yellow Sigatoka symptoms",
            "a photo of a diseased banana leaf with dark spots",
            "a photo of a healthy banana plant leaf",
            "a close-up photograph of a banana leaf in a plantation",
        ]

        self.negative_prompts = [
            "a photo of a mango leaf",
            "a photo of a coconut palm leaf",
            "a photo of a papaya leaf",
            "a photo of a leaf from an unrelated tropical plant",
            "a photo of a person",
            "a photo of an animal",
            "a photo of a building",
            "a photo of a landscape without a clearly visible banana leaf",
            "a photo of an unrelated object",
        ]

        self.candidate_labels = (
            self.positive_prompts + self.negative_prompts
        )

        self.banana_indices = list(
            range(len(self.positive_prompts))
        )

        print("CLIP banana-leaf gatekeeper initialized.")
        print("Device:", self.device)
        print("Model:", model_name)

    def predict(self, image_path):

        image_path = Path(image_path)

        validate_image(image_path)

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Could not process image: {image_path}"
            ) from exc

        inputs = self.processor(
            text=self.candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Normalized scores across the supplied descriptions.
            scores = outputs.logits_per_image.softmax(
                dim=1
            )[0]

        score_values = scores.cpu().tolist()

        positive_scores = [
            score_values[index]
            for index in self.banana_indices
        ]

        negative_scores = [
            score_values[index]
            for index in range(
                len(self.positive_prompts),
                len(self.candidate_labels),
            )
        ]

        banana_score = max(positive_scores)
        negative_score = max(negative_scores)

        best_index = int(scores.argmax().item())

        if (
            best_index in self.banana_indices
            and banana_score >= self.threshold
        ):
            status = "accepted_for_disease_screening"
            is_banana_leaf_candidate = True

        elif banana_score > negative_score:
            status = "uncertain"
            is_banana_leaf_candidate = False

        else:
            status = "rejected_or_uncertain"
            is_banana_leaf_candidate = False

        return {
            "is_banana_leaf_candidate": (
                is_banana_leaf_candidate
            ),
            "banana_leaf_score": float(banana_score),
            "strongest_negative_score": float(negative_score),
            "best_matching_description": (
                self.candidate_labels[best_index]
            ),
            "description_scores": {
                label: float(score)
                for label, score in zip(
                    self.candidate_labels,
                    score_values,
                )
            },
            "status": status,
        }
