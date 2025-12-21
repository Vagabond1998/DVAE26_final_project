from steps.ingest import download_mnist
from src.data import get_dataloaders


def test_mnist_train_batch_shape_and_labels(tmp_path):
    # Arrange: download into an isolated temp directory
    data_dir = tmp_path / "data"
    download_mnist(data_dir=str(data_dir))

    # Act: load one batch
    train_loader, _ = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=32,
        num_workers=0,  # avoids multiprocessing issues in CI/WSL
    )
    images, labels = next(iter(train_loader))

    # Assert: MNIST is grayscale 28x28, labels are digits 0..9
    assert images.shape == (32, 1, 28, 28)
    assert labels.shape == (32,)
    assert labels.dtype is not None
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 9

