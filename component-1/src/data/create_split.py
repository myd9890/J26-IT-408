from pathlib import Path
import random
import shutil

RANDOM_SEED = 42
TRAIN_PER_CLASS = 80

RAW_TRAIN = Path("data/raw/train")
PROCESSED = Path("data/processed")

CLASSES = [
    "healthy",
    "black_sigatoka",
    "yellow_sigatoka",
]

random.seed(RANDOM_SEED)

for class_name in CLASSES:

    source_dir = RAW_TRAIN / class_name

    images = sorted(source_dir.glob("*.jpg"))

    random.shuffle(images)

    train_images = images[:TRAIN_PER_CLASS]
    validation_images = images[TRAIN_PER_CLASS:]

    train_dir = PROCESSED / "train" / class_name
    validation_dir = PROCESSED / "validation" / class_name

    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy2(
            image,
            train_dir / image.name
        )

    for image in validation_images:
        shutil.copy2(
            image,
            validation_dir / image.name
        )

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(validation_images)} validation"
    )

print("\nDataset split completed.")