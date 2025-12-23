# src/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Confusion matrix with rows=true class, cols=predicted class.
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D tensors.")
    if y_true.numel() != y_pred.numel():
        raise ValueError("y_true and y_pred must have the same length.")

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.numel() == 0:
        return 0.0
    return (y_true == y_pred).float().mean().item()


def precision_recall_f1_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    """
    Compute macro-averaged precision, recall, f1 from a confusion matrix.
    Also returns per-class precision/recall/f1.
    """
    num_classes = cm.size(0)
    per_class = []

    for k in range(num_classes):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class.append({"precision": prec, "recall": rec, "f1": f1})

    macro_precision = sum(d["precision"] for d in per_class) / num_classes
    macro_recall = sum(d["recall"] for d in per_class) / num_classes
    macro_f1 = sum(d["f1"] for d in per_class) / num_classes

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
