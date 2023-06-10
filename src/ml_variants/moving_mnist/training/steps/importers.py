from zenml.steps import Output
from zenml import step
from satpy import Scene
import numpy as np
import sys
import typed_settings as ts
from torchvision.datasets import MovingMNIST
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

sys.path.append("../")

sys.path.append("../../../../")
sys.path.append("../../../../lib/data_loaders")

from Settings import ModelSettings
import lightning.pytorch as pl

settings = ts.load(ModelSettings, "model", ["config.toml"])

PATH_TO_DATA = "../../../../../data"


class MovingMnistModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transforms = transforms.Compose([])

    def prepare_data(self) -> None:
        MovingMNIST(self.data_dir, "train", download=True)
        MovingMNIST(self.data_dir, "test", download=True)

    def setup(self, stage: str):
        mnist_full = MovingMNIST(
            self.data_dir, split="train", transform=self.transforms
        )
        print(len(mnist_full))
        self.mnist_train, self.mnist_val = random_split(mnist_full, [8000, 2000])
        self.mnist_test = MovingMNIST(
            self.data_dir, split="test", transform=self.transforms
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, self.batch_size)


@step(enable_cache=False)
def importer_sat2rad() -> (
    Output(
        train_dataloader=DataLoader,
        val_dataloader=DataLoader,
        test_dataloader=DataLoader,
        predict_dataloader=DataLoader,
    )
):
    data_module = MovingMnistModule(
        data_dir=PATH_TO_DATA,
        batch_size=32,
    )
    data_module.prepare_data()
    data_module.setup(None)
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
        data_module.predict_dataloader(),
    )
