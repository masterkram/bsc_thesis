from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch
from enum import Enum
from zenml.steps import Output, step
import lightning.pytorch as pl
from satpy import Scene
import numpy as np


# class Sat2RadDataset(torch.utils.data.Dataset):
#     "Characterizes a dataset for PyTorch"

#     def __init__(
#         self,
#         path: str,
#         batch_size: int,
#         block_size: int,
#     ):
#         "Initialization"
#         self.path = path
#         self.batch_size = batch_size
#         self.block_size = block_size

#     def __len__(self):
#         "Denotes the total number of samples"
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         "Generates one sample of data"
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         X = torch.load("data/" + ID + ".pt")
#         y = self.labels[ID]

#         return X, y


class Sat2RadDataset(datasets.VisionDataset):
    def __len__(self):
        "Denotes the total number of samples"
        return 1

    def __getitem__(self, index):
        "Generates one sample of data"
        radar = np.load(
            "../../../../../data/preprocessed/radar/radar_nl_202304211800.npy"
        )
        satellite = np.load(
            "../../../../..//data/preprocessed/satellite/MSG3-SEVI-MSG15-0100-NA-20230421181241.751000000Z-NA.nat.npy"
        )

        satellite = np.reshape(satellite, (1, 4, 3712, 3712))

        X = torch.from_numpy(satellite)
        y = torch.from_numpy(radar)

        return X, y


class Sat2RadDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.train = Sat2RadDataset(
            self.data_dir, train=True, transforms=self.transform
        )
        # mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        # self.mnist_test = datasets.MNIST(
        #     self.data_dir, train=False, transform=self.transform
        # )
        # self.mnist_predict = datasets.MNIST(
        #     self.data_dir, train=False, transform=self.transform
        # )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)


@step
def importer_mnist() -> (
    Output(
        train_dataloader=DataLoader,
        test_dataloader=DataLoader,
    )
):
    data_module = Sat2RadDataModule(data_dir="../../../../../data", batch_size=32)
    data_module.prepare_data()
    data_module.setup(None)
    return (
        data_module.train_dataloader,
        data_module.test_dataloader,
        data_module.predict_dataloader,
    )


if __name__ == "__main__":
    radar = np.load("../../../../../data/preprocessed/radar/radar_nl_202304211800.npy")
    satellite = np.load(
        "../../../../..//data/preprocessed/satellite/MSG3-SEVI-MSG15-0100-NA-20230421181241.751000000Z-NA.nat.npy"
    )

    satellite = np.reshape(satellite, (1, 4, 3712, 3712))

    X = torch.from_numpy(satellite)
    y = torch.from_numpy(radar)

    print(X.size())
    print(y.size())
