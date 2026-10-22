from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "as-cle-bert/banana-disease-classification"

OUTPUT_DIR = Path("data/raw")

CLASS_MAP = {
    "banana_healthy_leaf": "healthy",
    "black_sigatoka": "black_sigatoka",
    "yellow_sigatoka": "yellow_sigatoka",
}

print("Loading dataset...")

train_dataset = load_dataset(DATASET_NAME, split="train")
test_dataset = load_dataset(DATASET_NAME, split="test")

print(f"Training images: {len(train_dataset)}")
print(f"Testing images: {len(test_dataset)}")

for split_name, dataset in [
    ("train", train_dataset),
    ("test", test_dataset),
]:
    for index, item in enumerate(dataset):

        label = dataset.features["label"].int2str(item["label"])

        if label not in CLASS_MAP:
            continue

        class_name = CLASS_MAP[label]

        output_folder = OUTPUT_DIR / split_name / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        image = item["image"]

        output_path = output_folder / f"{index:05d}.jpg"

        image.convert("RGB").save(output_path)

print("Dataset extraction completed.")