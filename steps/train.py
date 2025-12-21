# steps/train.py
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders
from src.model import MNISTCNN

_logger = logging.getLogger(__name__)


def train(
    data_dir: str = "./data",
    artifacts_dir: str = "./artifacts",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 2,
) -> Path:
    """
    Train MNIST CNN and save model weights to artifacts_dir.

    Returns the path to the saved model weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Device: %s", device)

    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # quick eval each epoch
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total

        _logger.info(
            "Epoch %d/%d - train_loss=%.4f train_acc=%.4f test_acc=%.4f",
            epoch, epochs, train_loss, train_acc, test_acc
        )

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_path / "mnist_cnn.pt"
    torch.save(model.state_dict(), out_path)
    _logger.info("Saved model weights to: %s", out_path)

    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()

