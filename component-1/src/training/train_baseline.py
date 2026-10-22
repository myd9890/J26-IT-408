import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from src.data.dataset import create_dataloaders
from src.models.baseline import create_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 16

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train():

    print("Using device:", DEVICE)

    train_loader, validation_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = create_model()
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):

        # -----------------
        # Training
        # -----------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # -----------------
        # Validation
        # -----------------

        model.eval()

        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                validation_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        validation_loss /= len(validation_loader)
        validation_accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        # -----------------
        # Save best model
        # -----------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                MODEL_DIR / "resnet18_baseline.pth"
            )

            print("  → Best model saved.")


if __name__ == "__main__":
    train()