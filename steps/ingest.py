import logging
from torchvision import datasets

_logger = logging.getLogger(__name__)

def download_mnist(data_dir):
   
    """
    Download the MNIST dataset if it is not already present.

    This function represents the data ingestion stage of the pipeline.
    It ensures that the MNIST dataset is available locally and verified.

    :param data_dir: Root directory where the MNIST dataset will be stored
    """
    _logger.info("Starting MNIST dataset ingestion.")
    _logger.info("Data directory: %s", data_dir)

    # Training set
    _logger.info("Checking / downloading MNIST training set...")
    datasets.MNIST(
            root=data_dir,
            train=True,
            download=True

    )

    # Test set
    _logger.info("Checking / downloading MNIST test set...")
    datasets.MNIST(
            root=data_dir,
            train=False,
            download=True
    )

    _logger.info("MNIST dataset ingestion completed successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_mnist()
