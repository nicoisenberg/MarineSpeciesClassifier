
from torch.utils.data import DataLoader
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


def create_dataloader(dataset, batch_size):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataloader
    