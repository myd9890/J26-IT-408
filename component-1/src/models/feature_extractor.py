import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


def create_feature_extractor():

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Remove the final classification layer.
    # The output will be a 512-dimensional feature vector.
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    return feature_extractor