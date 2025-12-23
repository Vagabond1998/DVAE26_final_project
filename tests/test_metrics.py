import torch
from src.metrics import confusion_matrix, accuracy, precision_recall_f1_from_cm


def test_confusion_matrix_simple():
    y_true = torch.tensor([0, 0, 1, 1, 2])
    y_pred = torch.tensor([0, 1, 1, 2, 2])
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.tolist() == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]


def test_accuracy_simple():
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.tensor([0, 2, 2, 3])
    assert abs(accuracy(y_true, y_pred) - 0.75) < 1e-9


def test_precision_recall_f1_from_cm_bounds():
    cm = torch.tensor([
        [5, 0],
        [1, 4],
    ])
    prf = precision_recall_f1_from_cm(cm)
    assert 0.0 <= prf["macro_precision"] <= 1.0
    assert 0.0 <= prf["macro_recall"] <= 1.0
    assert 0.0 <= prf["macro_f1"] <= 1.0
