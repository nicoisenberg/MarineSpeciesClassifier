
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def create_dataset(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(
        path,
        transform=transform
    )

    return dataset


def create_dataloader(dataset, batch_size, shuffle):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader


def split_dataset(dataset, training_percentage, validation_percentage):
    if training_percentage + validation_percentage >= 1:
        raise ValueError("Training and validation percentages must sum to less than 1.")

    training_size = int(len(dataset) * training_percentage)
    validation_size = int(len(dataset) * validation_percentage)
    test_size = len(dataset) - training_size - validation_size

    if training_size < 1 or validation_size < 1 or test_size < 1:
        raise ValueError("Each dataset split must contain at least one image.")

    generator = torch.Generator()
    generator.manual_seed(42)

    training_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [training_size, validation_size, test_size],
        generator=generator
    )

    return training_dataset, validation_dataset, test_dataset
    