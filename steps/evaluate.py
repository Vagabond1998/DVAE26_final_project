# steps/evaluate.py
import json
import logging
from pathlib import Path

import torch

from src.data import get_dataloaders
from src.model import MNISTCNN
from src.metrics import confusion_matrix, accuracy, precision_recall_f1_from_cm

_logger = logging.getLogger(__name__)

def evaluate(
    data_dir: str = "./data",
    artifacts_dir: str = "./artifacts",
    weights_path: str = "./artifacts/mnist_cnn.pt",
    batch_size: int = 128,
    num_workers: int = 2,
    num_classes: int = 10,
):
    """
    Evaluate trained MNIST CNN. Saves:
      - artifacts/metrics.json
      - artifacts/confusion_matrix.pt
      - artifacts/confusion_matrix.png
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Device: %s", device)

    _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = MNISTCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)

    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    prf = precision_recall_f1_from_cm(cm)

    metrics = {
        "test_accuracy": acc,
        "test_macro_precision": prf["macro_precision"],
        "test_macro_recall": prf["macro_recall"],
        "test_macro_f1": prf["macro_f1"],
        "example_count": int(y_true.numel()),
    }

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = artifacts_path / "confusion_matrix.pt"
    torch.save(cm, cm_path)

    # Save a simple confusion matrix image (matplotlib)
    try:
        import matplotlib.pyplot as plt  # allowed runtime dependency; if missing, we still keep .pt
        plt.figure()
        plt.imshow(cm.numpy())
        plt.title("MNIST Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        png_path = artifacts_path / "confusion_matrix.png"
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()
    except Exception as e:
        _logger.warning("Could not save confusion_matrix.png (matplotlib missing or failed): %s", e)

    _logger.info("Saved metrics to: %s", metrics_path)
    _logger.info("Saved confusion matrix to: %s", cm_path)

    return metrics_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate()

