from .dataset import create_datasets


train_dataset, validation_dataset, test_dataset = create_datasets()

print("Classes:", train_dataset.classes)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)