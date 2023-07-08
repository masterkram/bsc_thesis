import lightning.pytorch as pl
import torch
import os
import io
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader

mydatamodule = pl.LightningDataModule()


class LitModule(pl.LightningDataModule):
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
        if stage == "fit":
            mnist_full = datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
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

    def state_dict(self):
        # track whatever you want here
        state = {"batch_size": self.batch_size}
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.batch_size = state_dict["batch_size"]


with io.open(os.path.join(os.curdir, "train.pt"), "wb") as f:
    torch.save(mydatamodule, f)

with io.open(os.path.join(os.curdir, "train.pt"), "rb") as f:
    result = torch.load(mydatamodule, f)
    print(result)
