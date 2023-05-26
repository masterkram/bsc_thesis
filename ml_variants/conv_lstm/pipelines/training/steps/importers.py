from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch
from enum import Enum
from zenml.steps import Output, step
import pytorch_lightning as pl
from satpy import Scene
import numpy as np
import os

from utils.parse_time import order_based_on_file_timestamp
import math
import sys

sys.path.append("../")
sys.path.append("../../../../")

PATH_TO_DATA = "../../../../data"


class DatasetDistributor:
    def __init__(self, file_quantity: int, splits: list):
        self.file_quantity = file_quantity
        self.splits = splits
        self.index = 0
        self.distribution_position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.splits) <= self.index:
            raise StopIteration(
                f"not enough split parameters expected {self.index} but was {len(self.splits)}"
            )

        increase = math.floor(self.file_quantity * self.splits[self.index])
        upper_limit = self.distribution_position + increase

        if upper_limit > self.file_quantity:
            raise StopIteration(
                f"not enough files to satisfy split, wanted to allocate until {upper_limit}, but file amount is {self.file_quantity}"
            )

        lower_limit = self.distribution_position
        self.distribution_position = upper_limit
        self.index += 1

        return (lower_limit, upper_limit)


# class MyDataset:
#     def __init__(self):
#         self.dataset_distributor = DatasetDistributor()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         value = next(self.dataset_distributor)


class Sat2RadDataset(datasets.VisionDataset):
    def __init__(
        self,
        satellite_files: list[str],
        radar_files: list[str],
        root,
        transforms,
        radar_resolution_minutes: int = 5,
        satellite_seq_len: int = 5,
        radar_seq_len: int = 12,
    ):
        super().__init__(root, transform=transforms)
        self.satellite_files = satellite_files
        self.radar_files = radar_files
        self.satellite_seq_len = satellite_seq_len
        self.sat_len = len(self.satellite_files)
        self.rad_len = len(self.radar_files)
        self.radar_resolution_minutes = radar_resolution_minutes
        self.satellite_seq_len = satellite_seq_len
        self.radar_seq_len = radar_seq_len

    def __len__(self) -> int:
        "Denotes the total number of samples"
        seq_amount_sat = self.sat_len // 5
        seq_amount_rad = (self.rad_len - 5 * 3 - 4) // 15
        return min(seq_amount_sat, seq_amount_rad)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        "Generates one sample of data"

        lower_bound_satellite = index * self.satellite_seq_len
        upper_bound_satellite = (index + 1) * self.satellite_seq_len

        # beginning of the day has 2 extra images before satellite image
        lower_bound_radar = upper_bound_satellite * 3 + 2
        upper_bound_radar = lower_bound_radar + self.radar_seq_len

        satellite_sequence = [
            np.load(file)
            for file in self.satellite_files[
                lower_bound_satellite:upper_bound_satellite
            ]
        ]
        radar_sequence = [
            np.load(file)
            for file in self.radar_files[lower_bound_radar:upper_bound_radar]
        ]

        satellite_sequence = np.array(satellite_sequence)
        radar_sequence = np.array(radar_sequence)

        X = torch.from_numpy(satellite_sequence)
        y = torch.from_numpy(radar_sequence)

        return X, y.float()


def get_files_in_range(files: list, r: tuple[int, int]) -> list[str]:
    result = []
    for i in range(r[0], r[1]):
        result.append(files[i])
    return result


class Sat2RadDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        block_size: int = 5,
        batch_size: int = 32,
        splits: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.block_size = block_size
        self.splits = splits
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def load_with_full_path(self, path: str):
        load_dir = f"{self.data_dir}/preprocessed/{path}"
        files = os.listdir(load_dir)
        return list(map(lambda x: f"{load_dir}/{x}", files))

    def setup(self, stage: str):
        sat = order_based_on_file_timestamp(self.load_with_full_path("satellite"))
        rad = order_based_on_file_timestamp(self.load_with_full_path("radar"))

        amount_of_satellite_files = len(sat)
        amount_of_radar_files = len(rad)

        satdst = DatasetDistributor(
            file_quantity=amount_of_satellite_files, splits=list(self.splits.values())
        )
        raddst = DatasetDistributor(
            file_quantity=amount_of_radar_files, splits=list(self.splits.values())
        )

        train_satellite_files = get_files_in_range(sat, next(satdst))
        train_radar_files = get_files_in_range(rad, next(raddst))

        val_satellite_files = get_files_in_range(sat, next(satdst))
        val_radar_files = get_files_in_range(rad, next(raddst))

        test_satellite_files = get_files_in_range(sat, next(satdst))
        test_radar_files = get_files_in_range(rad, next(raddst))

        # training datasets
        self.train = Sat2RadDataset(
            satellite_files=train_satellite_files,
            radar_files=train_radar_files,
            root="",
            transforms=[],
        )
        self.validate = Sat2RadDataset(
            satellite_files=val_satellite_files,
            radar_files=val_radar_files,
            root="",
            transforms=[],
        )
        # test dataset
        self.test = Sat2RadDataset(
            satellite_files=test_satellite_files,
            radar_files=test_radar_files,
            root="",
            transforms=[],
        )
        # prediction dataset
        self.predict = Sat2RadDataset(
            satellite_files=test_satellite_files,
            radar_files=test_radar_files,
            root="",
            transforms=[],
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)


@step(enable_cache=False)
def importer_sat2rad() -> (
    Output(
        train_dataloader=DataLoader,
        test_dataloader=DataLoader,
        predict_dataloader=DataLoader,
    )
):
    data_module = Sat2RadDataModule(data_dir=PATH_TO_DATA, batch_size=1)
    data_module.prepare_data()
    data_module.setup(None)
    return (
        data_module.train_dataloader(),
        data_module.test_dataloader(),
        data_module.predict_dataloader(),
    )


if __name__ == "__main__":
    data_module = Sat2RadDataModule(data_dir=PATH_TO_DATA, batch_size=1)
    data_module.prepare_data()
    data_module.setup(None)
    dataset = data_module.train_dataloader()
    print(next(dataset)[1].size())
