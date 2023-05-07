from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from enum import Enum
from zenml.steps import Output, step


class Partition(Enum):
    train = "train"
    validation = "valid"
    test = "test"


partition_definition = {
    Partition.train: 0.8,
    Partition.validation: 0.1,
    Partition.test: 0.1,
}


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        path: str,
        batch_size: int,
        block_size: int,
        partition: Partition = Partition.train,
    ):
        "Initialization"
        self.path = path
        self.batch_size = batch_size
        self.block_size = block_size
        self.partition = partition

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load("data/" + ID + ".pt")
        y = self.labels[ID]

        return X, y


@step
def importer_mnist() -> (
    Output(
        train_dataloader=DataLoader,
        test_dataloader=DataLoader,
    )
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
