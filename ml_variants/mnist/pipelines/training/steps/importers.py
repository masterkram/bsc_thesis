from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from zenml.steps import Output, step
from zenml.materializers.base_materializer import BaseMaterializer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        normal_dist: list[int] = [0.1307, 0.3081],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((normal_dist[0],), (normal_dist[1],)),
            ]
        )

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = datasets.MNIST(
            self.data_dir, train=False, transform=self.transform
        )
        self.mnist_predict = datasets.MNIST(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


@step()
def importer_mnist() -> (
    Output(
        train_dataloader=DataLoader,
        test_dataloader=DataLoader,
        predict_dataloader=DataLoader,
    )
):
    """
    Import and partition MNIST dataset.

    Returns:
    + output (`tuple[DataLoader, DataLoader]`): Training Partition DataLoader and Testing Partition DataLoader
    """
    data_module = MNISTDataModule(batch_size=32)
    data_module.prepare_data()
    data_module.setup(None)
    return (
        data_module.train_dataloader(),
        data_module.test_dataloader(),
        data_module.predict_dataloader(),
    )
