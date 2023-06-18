from Sat2RadDataset import Sat2RadDataset
from util.parse_time import find_matching_string
import numpy as np


class SlidingDataset(Sat2RadDataset):
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

        self.length = len(
            np.lib.stride_tricks.sliding_window_view(
                self.satellite_files[0 : lastInput - 5], self.satellite_seq_len, axis=0
            )
        )

    def __len___(self):
        """denotes the total amount of samples"""
        return self.length
