from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset statistics (computed on MNIST training set)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Data augmentation parameters
MAX_ROTATION_DEG = 10


def get_transforms(train: bool = True):
    """
    Return torchvision transforms for MNIST.
    Training data uses augmentation; test data does not.
    """
    if train:
        return transforms.Compose([
            transforms.RandomRotation(MAX_ROTATION_DEG),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
):
    """
    Create DataLoaders for MNIST training and test sets.

    Assumes the dataset has already been downloaded by the ingest step.
    """
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
