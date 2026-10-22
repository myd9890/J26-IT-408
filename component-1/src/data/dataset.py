from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .transforms import (
    train_transforms,
    validation_transforms,
    test_transforms,
)


DATA_ROOT = Path("data")


def create_datasets():

    train_dataset = ImageFolder(
        DATA_ROOT / "processed" / "train",
        transform=train_transforms
    )

    validation_dataset = ImageFolder(
        DATA_ROOT / "processed" / "validation",
        transform=validation_transforms
    )

    test_dataset = ImageFolder(
        DATA_ROOT / "raw" / "test",
        transform=test_transforms
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def create_dataloaders(batch_size=16):

    train_dataset, validation_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )