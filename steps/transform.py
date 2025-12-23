from torchvision import transforms

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
