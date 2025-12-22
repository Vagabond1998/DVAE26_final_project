import json

from steps.ingest import download_mnist
from steps.train import train
from steps.evaluate import evaluate


def test_evaluate_runs_and_writes_metrics(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    weights_path = artifacts_dir / "mnist_cnn.pt"

    # Arrange
    download_mnist(data_dir=str(data_dir))
    train(
        data_dir=str(data_dir),
        artifacts_dir=str(artifacts_dir),
        epochs=1,
        batch_size=64,
        num_workers=0,
        lr=1e-3,
    )

    # Act
    metrics_path = evaluate(
        data_dir=str(data_dir),
        artifacts_dir=str(artifacts_dir),
        weights_path=str(weights_path),
        batch_size=128,
        num_workers=0,
    )

    # Assert
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "test_accuracy" in metrics
    assert "test_macro_precision" in metrics
    assert "test_macro_recall" in metrics
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
