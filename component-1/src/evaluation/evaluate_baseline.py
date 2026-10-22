import torch
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from src.data.dataset import create_datasets
from src.models.baseline import create_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("models/resnet18_baseline.pth")


def evaluate():

    print("Using device:", DEVICE)

    _, _, test_dataset = create_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("\n========== BASELINE TEST RESULTS ==========")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes,
            digits=4
        )
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(cm)


if __name__ == "__main__":
    evaluate()