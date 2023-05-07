from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from zenml.steps import Output, step


@step
def importer_mnist() -> Output(
    train_dataloader=DataLoader,
    test_dataloader=DataLoader,
):
    """Download the Fashion MNIST dataset."""
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 32

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader
