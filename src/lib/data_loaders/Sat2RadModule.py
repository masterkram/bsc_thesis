import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from typing import Type
from lib.data_loaders.DatasetType import DatasetType
from DatasetSlidingWindow import Sat2RadDatasetSlidingWindow
from DatasetDistributor import DatasetDistributorCombined
from Sat2RadDataset import Sat2RadDataset
from DatasetSequence import Sat2RadDatasetSequence
from ClassDatasetSequence import ClassDatasetSequence
from ClassSlidingSequence import ClassDatasetSlidingWindow
from typing import List, Dict, Tuple
from rich.table import Table

from util.parse_time import order_based_on_file_timestamp, parseTime, find_matching_time
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
        dataset_type: DatasetType = DatasetType.Sequence,
        regression=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.sequence_len_satellite = sequence_len_satellite
        self.sequence_len_radar = sequence_len_radar
        self.splits = splits
        self.transform = transforms.Compose([])
        print(dataset_type)
        self.dataset_type = dataset_type
        self.regression = regression

    def load_with_full_path(self, path: str):
        load_dir = f"{self.data_dir}/preprocessed/{path}"
        files = os.listdir(load_dir)
        return list(map(lambda x: f"{load_dir}/{x}", files))

    def ordered_files(self):
        radarPath = "radar" if self.regression else "radar-binned"

        sat = order_based_on_file_timestamp(self.load_with_full_path("satellite"))
        rad = order_based_on_file_timestamp(self.load_with_full_path(radarPath))

        # remove extra files from satellite:
        lastRadar = parseTime(rad[-1])
        lastSatellite = parseTime(sat[-1])

        lastDate = min(lastRadar, lastSatellite)
        lastTimeMatch = find_matching_time(sat, lastDate)

        sat = sat[0:lastTimeMatch]

        return sat, rad

    def get_dataset(self) -> Type[Sat2RadDataset]:
        if self.dataset_type == DatasetType.Sequence:
            return Sat2RadDatasetSequence
        elif self.dataset_type == DatasetType.ClassSlidingWindow:
            return ClassDatasetSlidingWindow
        elif self.dataset_type == DatasetType.SlidingWindow:
            return Sat2RadDatasetSlidingWindow

        return ClassDatasetSequence

    def setup(self, stage: str):
        self.sat, self.rad = self.ordered_files()

        amount_of_satellite_files = len(self.sat)
        amount_of_radar_files = len(self.rad)

        combinedDatasetDistributor = DatasetDistributorCombined(
            amount_of_satellite_files,
            amount_of_radar_files,
            splits=list(self.splits.values()),
        )

        train = next(combinedDatasetDistributor)
        val = next(combinedDatasetDistributor)
        test = next(combinedDatasetDistributor)

        self.train_satellite_files = get_files_in_range(self.sat, train[0])
        self.train_radar_files = get_files_in_range(self.rad, train[1])
        self.val_satellite_files = get_files_in_range(self.sat, val[0])
        self.val_radar_files = get_files_in_range(self.rad, val[1])
        self.test_satellite_files = get_files_in_range(self.sat, test[0])
        self.test_radar_files = get_files_in_range(self.rad, test[1])

        write_log(self.table(train, val, test))

        DataSet = self.get_dataset()

        write_log(f"using the dataset: {DataSet}")

        # training datasets
        self.train = DataSet(
            satellite_files=self.train_satellite_files,
            radar_files=self.train_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )
        self.validate = DataSet(
            satellite_files=self.val_satellite_files,
            radar_files=self.val_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )
        # test dataset
        self.test = DataSet(
            satellite_files=self.test_satellite_files,
            radar_files=self.test_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )
        # prediction dataset
        self.predict = DataSet(
            satellite_files=self.test_satellite_files,
            radar_files=self.test_radar_files,
            satellite_seq_len=self.sequence_len_satellite,
            radar_seq_len=self.sequence_len_radar,
        )

    def table(
        self, train: Tuple[int, int], val: Tuple[int, int], test: Tuple[int, int]
    ) -> Table:
        table = Table(title="Partitioned Files")

        table.add_column("Start Date Sat", justify="right", style="cyan")
        table.add_column("End Date Sat", justify="right", style="cyan")
        table.add_column("Start Date Rad", justify="right", style="cyan")
        table.add_column("End Date Rad", justify="right", style="cyan")
        table.add_column("Partition", style="magenta")
        table.add_column("Files Satellite", justify="right", style="green")
        table.add_column("Files Radar", justify="right", style="green")

        table.add_row(
            str(parseTime(self.sat[0])),
            str(parseTime(self.sat[-1])),
            str(parseTime(self.rad[0])),
            str(parseTime(self.rad[-1])),
            "All",
            str(len(self.sat)),
            str(len(self.rad)),
        )

        table.add_row(
            str(parseTime(self.sat[train[0][0]])),
            str(parseTime(self.sat[train[0][1]])),
            str(parseTime(self.rad[train[1][0]])),
            str(parseTime(self.rad[train[1][1]])),
            "Training",
            str(len(self.sat[train[0][0] : train[0][1]])),
            str(len(self.rad[train[1][0] : train[1][1]])),
        )
        table.add_row(
            str(parseTime(self.sat[val[0][0]])),
            str(parseTime(self.sat[val[0][1]])),
            str(parseTime(self.rad[val[1][0]])),
            str(parseTime(self.rad[val[1][1]])),
            "Validation",
            str(len(self.sat[val[0][0] : val[0][1]])),
            str(len(self.rad[val[1][0] : val[1][1]])),
        )
        table.add_row(
            str(parseTime(self.sat[test[0][0]])),
            str(parseTime(self.sat[test[0][1]])),
            str(parseTime(self.rad[test[1][0]])),
            str(parseTime(self.rad[test[1][1]])),
            "Testing",
            str(len(self.sat[test[0][0] : test[0][1]])),
            str(len(self.rad[test[1][0] : test[1][1]])),
        )
        return table

    def get_files(self) -> Dict:
        return {
            "training": {
                "sat": self.train_satellite_files,
                "rad": self.train_radar_files,
            },
            "valid": {"sat": self.val_satellite_files, "rad": self.val_radar_files},
            "test": {"sat": self.test_satellite_files, "rad": self.test_radar_files},
        }

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=12,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validate, batch_size=self.batch_size, num_workers=12, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=12, drop_last=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict, batch_size=self.batch_size, num_workers=12, drop_last=True
        )
