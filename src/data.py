from torchvision import datasets
from torch.utils.data import DataLoader
from steps.transform import get_transforms


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
):
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=get_transforms(train=True),
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
