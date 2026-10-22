import numpy as np
import torch
import torch.nn.functional as F


def generate_gradcam(
    model,
    image_tensor,
    prototypes,
    target_class,
    device,
):
    """
    Generate Grad-CAM for a ResNet18 prototypical classifier.

    Args:
        model: Frozen ResNet18 feature extractor.
        image_tensor: Preprocessed image tensor [3, 224, 224].
        prototypes: Prototype tensor [num_classes, 512].
        target_class: Index of the class to explain.
        device: CPU or CUDA device.

    Returns:
        A normalized Grad-CAM heatmap with shape [224, 224].
    """

    model.eval()

    activations = {}
    gradients = {}

    # In this feature extractor:
    # model[-2] is layer4.
    target_layer = model[-2]

    def forward_hook(module, inputs, output):

        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.__setitem__(
                "value",
                grad,
            )
        )

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        # Enable gradients on the input even though
        # the feature extractor's parameters are frozen.
        image_batch = (
            image_tensor
            .unsqueeze(0)
            .to(device)
            .detach()
            .requires_grad_(True)
        )

        prototypes = prototypes.to(device)

        model.zero_grad(set_to_none=True)

        # Feature extraction.
        features = model(image_batch)
        features = features.flatten(start_dim=1)

        # Preserve the normalization used by the classifier.
        embeddings = F.normalize(
            features,
            p=2,
            dim=1,
        )

        # Calculate distances to the prototypes.
        distances = torch.cdist(
            embeddings,
            prototypes,
        )

        # Explain the selected class by using its negative
        # Euclidean distance as the target score.
        score = -distances[0, target_class]

        score.backward()

        feature_maps = activations["value"]
        feature_gradients = gradients["value"]

        # Calculate channel weights.
        weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Construct the Grad-CAM heatmap.
        cam = torch.relu(
            (weights * feature_maps).sum(
                dim=1,
                keepdim=True,
            )
        )

        # Resize the heatmap to the input dimensions.
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    finally:
        # Avoid accumulating hooks across inference calls.
        hook.remove()