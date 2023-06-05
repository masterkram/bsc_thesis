import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from typing import Type
from DatasetType import DatasetType
from DatasetSlidingWindow import Sat2RadDatasetSlidingWindow
from DatasetDistributor import DatasetDistributor
from Sat2RadDataset import Sat2RadDataset
from DatasetSequence import Sat2RadDatasetSequence
from rich import table

from util.parse_time import order_based_on_file_timestamp
from util.log_utils import write_log


def get_files_in_range(files: list, r: tuple[int, int]) -> list[str]:
    result = []
    for i in range(r[0], r[1]):
        result.append(files[i])
    return result


class Sat2RadDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        sequence_len_satellite: int = 5,
        sequence_len_radar: int = 12,
        batch_size: int = 1,
        splits: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.sequence_len_satellite = sequence_len_satellite
        self.sequence_len_radar = sequence_len_radar
        self.splits = splits
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.dataset_type = DatasetType.Sequence

    def load_with_full_path(self, path: str):
        load_dir = f"{self.data_dir}/preprocessed/{path}"
        files = os.listdir(load_dir)
        return list(map(lambda x: f"{load_dir}/{x}", files))

    def ordered_files(self):
        sat = order_based_on_file_timestamp(self.load_with_full_path("satellite"))
        rad = order_based_on_file_timestamp(self.load_with_full_path("radar"))
        return sat, rad

    def get_dataset(self) -> Type[Sat2RadDataset]:
        match self.dataset_type:
            case DatasetType.SlidingWindow:
                return Sat2RadDatasetSlidingWindow
            case _:
                return Sat2RadDatasetSequence

    def setup(self, stage: str):
        sat, rad = self.ordered_files()

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

        t = table.Table(title="Partitioned Files")
        t.add_column("split", style="magenta")
        t.add_column("satellite", justify="right", style="green")
        t.add_column("radar", justify="right", style="cyan")
        t.add_row(
            "training", str(len(train_satellite_files)), str(len(train_radar_files))
        )
        t.add_row(
            "validation", str(len(val_satellite_files)), str(len(val_radar_files))
        )
        t.add_row("testing", str(len(test_satellite_files)), str(len(test_radar_files)))

        write_log(t)

        DataSet = self.get_dataset()

        # training datasets
        self.train = DataSet(
            satellite_files=train_satellite_files,
            radar_files=train_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )
        self.validate = DataSet(
            satellite_files=val_satellite_files,
            radar_files=val_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )
        # test dataset
        self.test = DataSet(
            satellite_files=test_satellite_files,
            radar_files=test_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )
        # prediction dataset
        self.predict = DataSet(
            satellite_files=test_satellite_files,
            radar_files=test_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)
