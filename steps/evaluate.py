# steps/evaluate.py
import json
import logging
from pathlib import Path

import torch

from src.data import get_dataloaders
from src.model import MNISTCNN

_logger = logging.getLogger(__name__)


def _confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Compute confusion matrix as a (num_classes, num_classes) tensor.
    Rows: true class, Cols: predicted class.
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def _precision_recall_from_cm(cm: torch.Tensor):
    """
    Returns macro precision and macro recall from confusion matrix.
    """
    num_classes = cm.size(0)
    precisions = []
    recalls = []

    for k in range(num_classes):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    return macro_precision, macro_recall


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

    accuracy = (y_pred == y_true).float().mean().item()

    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    macro_precision, macro_recall = _precision_recall_from_cm(cm)

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "test_accuracy": accuracy,
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "example_count": int(y_true.numel()),
        "weights_path": str(Path(weights_path)),
        "device": str(device),
    }

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

