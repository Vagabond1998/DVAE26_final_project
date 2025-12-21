from steps.ingest import download_mnist
from steps.train import train


def test_train_runs_and_saves_model(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"

    download_mnist(data_dir=str(data_dir))
    out_path = train(
        data_dir=str(data_dir),
        artifacts_dir=str(artifacts_dir),
        epochs=1,
        batch_size=64,
        num_workers=0,
    )
    assert out_path.exists()
