import torch
import numpy as np
from Sat2RadDataset import Sat2RadDataset
import math
from util.parse_time import get_next_sequence, find_matching_string, parseTime
from datetime import datetime
from torchvision.transforms.functional import resize, InterpolationMode
from torchvision.transforms import AutoAugment
from util.log_utils import write_log


def loadFile(file: str):
    arr = np.load(file)
    return addTimeDim(arr, file)


def addTimeDim(array: np.ndarray, time: str):
    now = parseTime(time).hour / 24
    times = np.array([now]).repeat(256 * 256).reshape((1, 256, 256))
    return np.concatenate((array, times), axis=0)


class ClassDatasetSlidingWindow(Sat2RadDataset):
    def __init__(
        self,
        satellite_files: list[str],
        radar_files: list[str],
        satellite_seq_len: int = 5,
        radar_seq_len: int = 12,
    ):
        super().__init__(
            satellite_files,
            radar_files,
            "",
            [],
            satellite_seq_len,
            radar_seq_len,
            stride=True,
        )

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        "Generates one sample of data"

        lower_bound_satellite = index
        upper_bound_satellite = index + self.satellite_seq_len

        # beginning of the day has 2 extra images before satellite image
        lower_bound_radar, upper_bound_radar = get_next_sequence(
            self.radar_seq_len,
            self.satellite_files[upper_bound_satellite],
            self.radar_files,
        )

        satellite_sequence = [
            loadFile(file)
            for file in self.satellite_files[
                lower_bound_satellite:upper_bound_satellite
            ]
        ]
        # radar_sequence = [
        #    np.load(file)
        #   for file in self.radar_files[lower_bound_radar:upper_bound_radar]
        # ]

        satellite_sequence = np.array(satellite_sequence)
        radar_sequence = np.load(self.radar_files[upper_bound_radar])

        write_log(
            f"sat({parseTime(self.satellite_files[lower_bound_satellite])} - {parseTime(self.satellite_files[upper_bound_satellite])}): rad({parseTime(self.radar_files[lower_bound_radar])} - {parseTime(self.radar_files[upper_bound_radar])})"
        )

        X = torch.from_numpy(satellite_sequence)
        y = torch.from_numpy(radar_sequence).view(1, 1660, 1340)
        print(y.shape)
        y = resize(y, [256, 256], interpolation=InterpolationMode.NEAREST)

        return X.float(), y.long()
