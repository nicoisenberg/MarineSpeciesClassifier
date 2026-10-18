from src.dataset import create_dataloader, create_dataset


def main():
    dataset = create_dataset("./data")

    print(f"Class names: {dataset.classes}")
    print(f"Class indexes: {dataset.class_to_idx}")
    print(f"Number of images: {len(dataset)}")

    image_tensor, label = dataset[0]

    print(f"Shape: {image_tensor.shape}")
    print(f"Label: {label}")
    print(f"Class: {dataset.classes[label]}")

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=2,
    )

    batch_number = 1

    for images, labels in dataloader:
        print(f"Images shape of Batch {batch_number}: {images.shape}")
        print(f"Labels shape of Batch {batch_number}: {labels.shape}")
        print(f"Labels of batch {batch_number}: {labels}")
        batch_number += 1


if __name__ == "__main__":
    main()