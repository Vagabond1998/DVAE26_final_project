from pathlib import Path
from steps.ingest import download_mnist

def test_download_mnist(tmp_path):
    data_dir = tmp_path / "data"
    download_mnist(data_dir = str(data_dir))

    raw_dir = data_dir / "MNIST" / "raw"
    assert raw_dir.exists()
