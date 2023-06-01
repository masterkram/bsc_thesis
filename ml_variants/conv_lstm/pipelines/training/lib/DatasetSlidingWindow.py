import math
from Sat2RadDataset import Sat2RadDataset
import numpy as np
import torch


def length_sliding_window_satellite(file_number: int, sequence_length: int) -> int:
    return math.floor(
        sum([x / sequence_length for x in range(sequence_length, file_number)])
    )


def length_sliding_window_radar(
    file_number: int, sequence_length: int, sequence_length_satellite: int
):
    res = 0
    for i in range(0, 4):
        ls = ((file_number - 16) - 3 * i) // 12
        res += ls

    return res


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

        len_sat_sliding = length_sliding_window_satellite(
            self.sat_len, self.satellite_seq_len
        )
        len_rad_sliding = length_sliding_window_radar(
            self.rad_len, self.radar_seq_len, self.satellite_seq_len
        )

        print(f"calculating lengths, {len_sat_sliding} -> {len_rad_sliding}")

        self.len: int = min(len_sat_sliding, len_rad_sliding)

    def __len__(self) -> int:
        "Denotes the total number of samples"
        return self.len

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        "Generates one sample of data"

        lower_bound_satellite = index
        upper_bound_satellite = index + self.satellite_seq_len

        # beginning of the day has 2 extra images before satellite image
        lower_bound_radar = upper_bound_satellite * 3 + 1
        upper_bound_radar = lower_bound_radar + self.radar_seq_len

        assert upper_bound_radar - lower_bound_radar == self.radar_seq_len
        assert upper_bound_satellite - lower_bound_satellite == self.satellite_seq_len

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

        assert len(satellite_sequence) == self.satellite_seq_len
        assert len(radar_sequence) == self.radar_seq_len

        satellite_sequence = np.array(satellite_sequence)
        radar_sequence = np.array(radar_sequence)

        X = torch.from_numpy(satellite_sequence)
        y = torch.from_numpy(radar_sequence)

        return X, y.float()
