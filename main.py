from src.dataset import create_dataloader, create_dataset, split_dataset


def main():
    dataset = create_dataset("./data")

    print(f"Class names: {dataset.classes}")
    print(f"Class indexes: {dataset.class_to_idx}")
    print(f"Number of images: {len(dataset)}")

    training_dataset, validation_dataset, test_dataset = split_dataset(
        dataset,
        training_percentage=0.5,
        validation_percentage=0.25
    )

    print(f"Number of training images: {len(training_dataset)}")
    print(f"Number of validation images: {len(validation_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Training images indices: {training_dataset.indices}")
    print(f"Validation images indices: {validation_dataset.indices}")
    print(f"Test images indices: {test_dataset.indices}")

    training_dataloader = create_dataloader(
        training_dataset,
        batch_size=2,
        shuffle=True
    )
    validation_dataloader = create_dataloader(
        validation_dataset,
        batch_size=2,
        shuffle=False
    )
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=2,
        shuffle=False
    )

    print(f"Batches of the training dataloader: {len(training_dataloader)}")
    print(f"Batches of the validation dataloader: {len(validation_dataloader)}")
    print(f"Batches of the test dataloader: {len(test_dataloader)}")


if __name__ == "__main__":
    main()