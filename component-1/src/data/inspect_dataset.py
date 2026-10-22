from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path("data/raw")

CLASSES = [
    "healthy",
    "black_sigatoka",
    "yellow_sigatoka",
]

random.seed(42)

fig, axes = plt.subplots(
    3,
    5,
    figsize=(15, 9)
)

for row, class_name in enumerate(CLASSES):

    images = list((ROOT / "train" / class_name).glob("*.jpg"))

    selected = random.sample(images, min(5, len(images)))

    for col, image_path in enumerate(selected):

        image = Image.open(image_path)

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name.replace("_", " ").title())
        axes[row, col].axis("off")

plt.tight_layout()

output = Path("results")
output.mkdir(exist_ok=True)

plt.savefig(
    output / "dataset_visual_inspection.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()