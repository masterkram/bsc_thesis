import math
from Sat2RadDataset import Sat2RadDataset
import numpy as np
import torch
from util.parse_time import get_next_sequence, find_matching_string


class Sat2RadDatasetSlidingWindow(Sat2RadDataset):
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

        lastPrediction = self.radar_files[
            (self.rad_len - 1) - (self.radar_seq_len - 1) - 1
        ]
        lastInput = find_matching_string(self.satellite_files, lastPrediction)

        self.len_sat_sliding = len(
            np.lib.stride_tricks.sliding_window_view(
                self.satellite_files[0 : lastInput - 5], self.satellite_seq_len, axis=0
            )
        )

    def __len__(self) -> int:
        "Denotes the total number of samples"
        return self.len_sat_sliding

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

        return X.float(), y.float()
