import torch
import numpy as np
from Sat2RadDataset import Sat2RadDataset
import math
from util.parse_time import get_next_sequence, find_matching_string, parseTime
from datetime import datetime
from torchvision.transforms.functional import resize, InterpolationMode


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
            satellite_files, radar_files, "", [], satellite_seq_len, radar_seq_len
        )
        lastPrediction = self.radar_files[(self.rad_len - 1) - (self.radar_seq_len - 1)]
        lastInput = find_matching_string(self.satellite_files, lastPrediction)
        if lastInput is None:
            lastPrediction = self.radar_files[
                (self.rad_len - 1) - (self.radar_seq_len - 1) - 1
            ]
            lastInput = find_matching_string(self.satellite_files, lastPrediction)

        self.length = lastInput // self.satellite_seq_len - 1

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        "Generates one sample of data"

        lower_bound_satellite = index * self.satellite_seq_len
        upper_bound_satellite = (index + 1) * self.satellite_seq_len

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
        radar_sequence = [
            np.load(file)
            for file in self.radar_files[lower_bound_radar:upper_bound_radar]
        ]

        satellite_sequence = np.array(satellite_sequence)
        radar_sequence = np.array(radar_sequence)

        X = torch.from_numpy(satellite_sequence)
        y = torch.from_numpy(radar_sequence)
        y = resize(y, [256, 256], interpolation=InterpolationMode.NEAREST)

        return X.float(), y.long()
