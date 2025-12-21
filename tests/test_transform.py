import torch
from PIL import Image
import numpy as np

from src.data import get_transforms


def test_train_transform_outputs_tensor_with_correct_shape():
    transform = get_transforms(train=True)

    # Fake MNIST-like image: 28x28 grayscale
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))

    out = transform(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 28, 28)


def test_train_and_test_transforms_are_different():
    train_tf = get_transforms(train=True)
    test_tf = get_transforms(train=False)

    assert train_tf != test_tf

