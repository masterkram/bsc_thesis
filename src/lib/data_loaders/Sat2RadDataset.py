from torchvision import datasets
from util.parse_time import order_based_on_file_timestamp, find_matching_string
from typing import List
import numpy as np


def find_last_satellite_file(
    satellite_files: List, radar_files: List, rad_len: int, radar_seq_len: int
) -> int:
    """
    Find the index of the last working satellite image in the split dataset.
    """
    lastRadarIndex = (rad_len - 1) - (radar_seq_len - 1)
    lastPrediction = radar_files[lastRadarIndex]
    lastInput = find_matching_string(satellite_files, lastPrediction)

    if lastInput is not None:
        return lastInput

    # try next if not found.
    i = 1
    while i < 10 and lastInput is None:
        lastPrediction = radar_files[lastRadarIndex - i]
        lastInput = find_matching_string(satellite_files, lastPrediction)
        i += 1

    return lastInput


def length_sequence(lastInput: int, satellite_seq_len: int) -> int:
    return lastInput // satellite_seq_len - 1


def length_stride(satellite_files: List, lastInput: int, satellite_seq_len: int) -> int:
    return len(
        np.lib.stride_tricks.sliding_window_view(
            satellite_files[0 : lastInput - 5], satellite_seq_len, axis=0
        )
    )


class Sat2RadDataset(datasets.VisionDataset):
    def __init__(
        self,
        satellite_files: list[str],
        radar_files: list[str],
        root,
        transforms,
        satellite_seq_len: int = 5,
        radar_seq_len: int = 12,
        stride=False,
    ):
        super().__init__(root, transform=transforms)
        self.satellite_files = order_based_on_file_timestamp(satellite_files)
        self.radar_files = order_based_on_file_timestamp(radar_files)
        self.satellite_seq_len = satellite_seq_len
        self.sat_len = len(self.satellite_files)
        self.rad_len = len(self.radar_files)
        self.satellite_seq_len = satellite_seq_len
        self.radar_seq_len = radar_seq_len
        lastSatellite = find_last_satellite_file(
            satellite_files, radar_files, self.rad_len, self.radar_seq_len
        )
        self.length = (
            length_sequence(lastSatellite, satellite_seq_len)
            if not stride
            else length_stride(satellite_files, lastSatellite, self.satellite_seq_len)
        )

    def __len__(self):
        return self.length
