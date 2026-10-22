"""
Verification Gate 3: Semantic Segmentation & Morphometric Scope Engine Suite.

Validates:
1. Classical rule-based segmenter baseline (FR2).
2. PyTorch 3-class U-Net forward pass, backward pass, and mask prediction (FR2).
3. Combined Multi-Class Soft Dice + Cross-Entropy loss computation and gradient propagation.
4. Per-class and macro segmentation metric fidelity (IoU, Dice, Precision, Recall).
5. Morphometric Scope Engine (FR3) boundary conditions and feature extraction.
6. End-to-end integration: Image -> U-Net -> Mask -> MorphometricFeatures.
"""

import os
import pytest
import numpy as np
import torch

from src.segmentation.classical_baseline import ClassicalSegmenter
from src.segmentation.unet import UNet
from src.segmentation.loss_and_metrics import (
    SoftDiceLoss,
    CombinedDiceCrossEntropyLoss,
    compute_segmentation_metrics,
)
from src.segmentation.scope_calculator import (
    MorphometricFeatures,
    extract_leaf_and_lesion_features,
)
from src.segmentation.train_seg import train_unet


@pytest.fixture
def sample_rgb_image():
    """Generates a synthetic 256x256 RGB banana leaf sample with a necrotic patch."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Background is black
    # Leaf lamina is an ellipse with healthy green (R=50, G=160, B=40)
    import cv2
    cv2.ellipse(img, (128, 128), (100, 35), 45, 0, 360, (50, 160, 40), -1)
    # Lesion patch in the center: dark brown (R=70, G=35, B=20)
    cv2.ellipse(img, (128, 128), (20, 10), 45, 0, 360, (70, 35, 20), -1)
    return img


@pytest.fixture
def sample_ground_truth_mask():
    """Generates a matching 256x256 ground truth 3-class mask."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    import cv2
    # Leaf lamina = 1
    cv2.ellipse(mask, (128, 128), (100, 35), 45, 0, 360, 1, -1)
    # Lesion = 2
    cv2.ellipse(mask, (128, 128), (20, 10), 45, 0, 360, 2, -1)
    return mask


def test_classical_baseline(sample_rgb_image):
    """Verifies ClassicalSegmenter predicts valid 3-class masks."""
    segmenter = ClassicalSegmenter(min_lesion_pixels=5, target_size=(256, 256))
    mask = segmenter.predict(sample_rgb_image)

    assert mask.shape == (256, 256), f"Expected shape (256, 256), got {mask.shape}"
    assert mask.dtype == np.uint8
    unique_classes = set(np.unique(mask))
    assert unique_classes.issubset({0, 1, 2}), f"Unexpected classes: {unique_classes}"

    # Verify that leaf and lesion were detected on our sample image
    assert 1 in unique_classes, "Healthy leaf tissue (class 1) was not detected"
    assert 2 in unique_classes, "Lesion tissue (class 2) was not detected"

    # Deterministic test
    mask_2 = segmenter.predict(sample_rgb_image)
    assert np.array_equal(mask, mask_2), "Classical baseline must be strictly deterministic"


def test_unet_architecture():
    """Verifies U-Net tensor shapes for both batch and single input."""
    model = UNet(n_channels=3, n_classes=3, base_filters=16)
    model.eval()

    # Batch of 2 images: (B, 3, H, W)
    batch_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(batch_input)
    assert logits.shape == (2, 3, 256, 256), f"Unexpected logits shape: {logits.shape}"

    # Batch prediction
    preds = model.predict_mask(batch_input)
    assert preds.shape == (2, 256, 256), f"Unexpected preds shape: {preds.shape}"
    assert preds.dtype == torch.int64

    # Single image prediction: (3, H, W)
    single_input = torch.randn(3, 256, 256)
    single_pred = model.predict_mask(single_input)
    assert single_pred.shape == (256, 256), f"Unexpected single pred shape: {single_pred.shape}"


def test_loss_functions():
    """Verifies Dice and Combined loss calculation and backward pass."""
    logits = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randint(0, 3, (2, 64, 64), dtype=torch.int64)

    # SoftDiceLoss
    dice_loss_fn = SoftDiceLoss()
    d_loss = dice_loss_fn(logits, target)
    assert torch.isfinite(d_loss).item(), "Dice loss is not finite"
    assert d_loss.item() >= 0.0, "Dice loss should be non-negative"

    # CombinedDiceCrossEntropyLoss
    combined_fn = CombinedDiceCrossEntropyLoss(dice_weight=0.5, ce_weight=0.5)
    loss = combined_fn(logits, target)
    assert torch.isfinite(loss).item(), "Combined loss is not finite"
    assert loss.item() > 0.0, "Combined loss should be strictly positive"

    # Verify backpropagation
    loss.backward()
    assert logits.grad is not None, "Gradients failed to propagate to input logits"
    assert torch.all(torch.isfinite(logits.grad)), "Gradients contain NaN or Inf"


def test_segmentation_metrics(sample_ground_truth_mask):
    """Verifies accuracy, IoU, Dice, Precision, and Recall metric calculations."""
    gt = sample_ground_truth_mask

    # Perfect match test
    perfect_metrics = compute_segmentation_metrics(gt, gt)
    assert perfect_metrics["overall_accuracy"] == 1.0
    assert perfect_metrics["lesion_dice"] == 1.0
    assert perfect_metrics["leaf_dice"] == 1.0
    assert perfect_metrics["macro_dice"] == 1.0

    # Completely disjoint prediction (all background)
    empty_pred = np.zeros_like(gt)
    empty_metrics = compute_segmentation_metrics(empty_pred, gt)
    assert empty_metrics["lesion_dice"] < 0.05
    assert empty_metrics["lesion_recall"] == 0.0

    # Both empty masks (no division by zero)
    all_zero_1 = np.zeros((100, 100), dtype=np.uint8)
    all_zero_2 = np.zeros((100, 100), dtype=np.uint8)
    zero_metrics = compute_segmentation_metrics(all_zero_1, all_zero_2)
    assert zero_metrics["overall_accuracy"] == 1.0
    assert zero_metrics["macro_dice"] == 1.0


def test_morphometric_scope_engine(sample_ground_truth_mask):
    """Verifies FR3 Morphometric Scope Engine feature extraction and boundary constraints."""
    features = extract_leaf_and_lesion_features(sample_ground_truth_mask, min_lesion_pixels=5)

    assert isinstance(features, MorphometricFeatures)
    assert 0.0 <= features.scope_pct <= 100.0
    assert 0.0 <= features.coalescence_ratio_pct <= 100.0
    assert 0.0 <= features.morph_distortion_idx <= 100.0
    assert features.num_lesions >= 1
    assert features.leaf_pixels > features.lesion_pixels > 0

    # Edge Case 1: Empty mask (zero leaf pixels)
    empty_mask = np.zeros((256, 256), dtype=np.uint8)
    empty_feat = extract_leaf_and_lesion_features(empty_mask)
    assert empty_feat.scope_pct == 0.0
    assert empty_feat.num_lesions == 0
    assert empty_feat.coalescence_ratio_pct == 0.0
    assert empty_feat.leaf_pixels == 0
    assert empty_feat.morph_distortion_idx == 0.0

    # Edge Case 2: 100% Healthy leaf (class 1 everywhere inside leaf, no class 2)
    healthy_mask = np.zeros((256, 256), dtype=np.uint8)
    healthy_mask[50:200, 50:200] = 1
    healthy_feat = extract_leaf_and_lesion_features(healthy_mask)
    assert healthy_feat.scope_pct == 0.0
    assert healthy_feat.num_lesions == 0
    assert healthy_feat.coalescence_ratio_pct == 0.0
    assert healthy_feat.leaf_pixels == 150 * 150

    # Edge Case 3: 100% Lesion coverage (class 2 everywhere inside leaf)
    total_lesion_mask = np.zeros((256, 256), dtype=np.uint8)
    total_lesion_mask[50:200, 50:200] = 2
    full_feat = extract_leaf_and_lesion_features(total_lesion_mask)
    assert full_feat.scope_pct == 100.0
    assert full_feat.coalescence_ratio_pct == 100.0


def test_aspect_ratio_distortion_scaling():
    """Verifies lamina aspect ratio and bunching/distortion index calculation."""
    # Healthy slender lamina: 3.5 aspect ratio -> distortion should be ~0.0
    h_mask = np.zeros((350, 100), dtype=np.uint8)
    h_mask[:, :] = 1
    h_feat = extract_leaf_and_lesion_features(h_mask)
    assert abs(h_feat.leaf_aspect_ratio - 3.5) < 0.2
    assert h_feat.morph_distortion_idx < 10.0

    # Stunted bunched lamina: 1.5 aspect ratio -> distortion should be ~100.0
    s_mask = np.zeros((150, 100), dtype=np.uint8)
    s_mask[:, :] = 1
    s_feat = extract_leaf_and_lesion_features(s_mask)
    assert abs(s_feat.leaf_aspect_ratio - 1.5) < 0.2
    assert s_feat.morph_distortion_idx > 90.0


def test_quick_training_loop_integration(tmp_path):
    """
    Runs a fast 2-epoch training routine with U-Net on synthetic metadata
    to ensure full end-to-end integration and weights checkpoint generation.
    """
    weights_path = str(tmp_path / "test_unet_weights.pt")
    history_path = str(tmp_path / "test_history.json")

    history = train_unet(
        metadata_path="data/synthetic/metadata.csv",
        base_dir="data/synthetic",
        output_weights_path=weights_path,
        history_path=history_path,
        epochs=2,
        batch_size=8,
        learning_rate=0.002,
        base_filters=16,
        max_train_samples=24,
        max_val_samples=12
    )

    assert os.path.exists(weights_path), "U-Net checkpoint file was not created"
    assert os.path.exists(history_path), "Training history file was not created"
    assert len(history["epochs"]) == 2
    assert history["best_val_macro_dice"] > 0.0

    # Verify that the saved checkpoint can be loaded and perform inference
    checkpoint = torch.load(weights_path, map_location="cpu")
    model = UNet(n_channels=3, n_classes=3, base_filters=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_tensor = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        pred_mask = model.predict_mask(dummy_tensor).squeeze(0).numpy()

    features = extract_leaf_and_lesion_features(pred_mask)
    assert isinstance(features, MorphometricFeatures)
    assert 0.0 <= features.scope_pct <= 100.0
