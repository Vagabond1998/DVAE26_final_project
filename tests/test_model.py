import torch
from src.model import MNISTCNN


def test_mnistcnn_forward_shape():
    model = MNISTCNN()
    x = torch.randn(16, 1, 28, 28)
    y = model(x)
    assert y.shape == (16, 10)
