import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        print("Loading pretrained DINOv2 ViT-S/14...")

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14"
        )

        # Freeze the pretrained feature extractor.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.backbone.eval()

        print("DINOv2 loaded successfully.")

    def forward(self, images):

        # Extract features without calculating gradients.
        with torch.no_grad():
            embeddings = self.backbone(images)

        return embeddings


def create_dinov2_feature_extractor(device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = DINOv2FeatureExtractor()
    model = model.to(device)
    model.eval()

    return model

