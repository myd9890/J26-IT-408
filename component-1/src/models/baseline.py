import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 3


def create_model():
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Freeze the pretrained feature extractor
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the classification head
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model