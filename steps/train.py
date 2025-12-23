# steps/train.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import get_dataloaders
from src.model import MNISTCNN

_logger = logging.getLogger(__name__)

def _evaluate_loss_and_accuracy(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            bs = images.size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def train(
    data_dir: str = "./data",
    artifacts_dir: str = "./artifacts",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 2,
    seed: int = 42,
    save_best_only: bool = True,
) -> Path:
    """
    Full training for MNIST CNN.
    Saves:
      - artifacts/mnist_cnn.pt (best or last weights)
      - artifacts/train_history.json
    Returns path to saved weights.
    """
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Device: %s", device)
    if device.type == "cuda":
        _logger.info("GPU: %s", torch.cuda.get_device_name(0))

    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_epoch = 0
    best_test_loss = None
    best_path = artifacts_path / "mnist_cnn.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs

            train_loss = running_loss / total
            train_acc = correct / total
            pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = _evaluate_loss_and_accuracy(model, test_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        _logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
            epoch, epochs, train_loss, train_acc, test_loss, test_acc
        )

        # Save best checkpoint
        if (not save_best_only) or (test_acc > best_acc):
            best_epoch = epoch
            best_test_loss = test_loss
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            _logger.info("Saved checkpoint: %s (best_acc=%.4f)", best_path, best_acc)

    # Save training history
    hist_path = artifacts_path / "train_history.json"
    out = {
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "best_test_loss": best_test_loss,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "num_workers": num_workers,
            "seed": seed,
        },
        "history": history,
    }

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    _logger.info("Saved training history: %s", hist_path)

    return best_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
