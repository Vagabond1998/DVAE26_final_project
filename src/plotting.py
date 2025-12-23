# src/plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_history(history: Dict[str, List[float]], out_dir: str | Path = None):
    """
    history expects keys: 'train_loss', 'train_acc', 'test_acc'
    If out_dir is provided, saves PNGs there. Also returns figure objects.
    """
    figs = []

    # Loss
    fig1 = plt.figure()
    plt.plot(history["train_loss"])
    plt.title("Train Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    figs.append(fig1)

    # Accuracy
    fig2 = plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["test_acc"], label="test_acc")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    figs.append(fig2)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out_dir / "train_loss.png", bbox_inches="tight")
        fig2.savefig(out_dir / "accuracy.png", bbox_inches="tight")

    return figs
