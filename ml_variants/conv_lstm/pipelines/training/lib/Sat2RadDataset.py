from torchvision import datasets


class Sat2RadDataset(datasets.VisionDataset):
    def __init__(
        self,
        satellite_files: list[str],
        radar_files: list[str],
        root,
        transforms,
        satellite_seq_len: int = 5,
        radar_seq_len: int = 12,
    ):
        super().__init__(root, transform=transforms)
        self.satellite_files = satellite_files
        self.radar_files = radar_files
        self.satellite_seq_len = satellite_seq_len
        self.sat_len = len(self.satellite_files)
        self.rad_len = len(self.radar_files)
        self.satellite_seq_len = satellite_seq_len
        self.radar_seq_len = radar_seq_len
