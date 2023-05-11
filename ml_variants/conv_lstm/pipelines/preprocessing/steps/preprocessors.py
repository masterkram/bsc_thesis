from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from enum import Enum
from zenml.steps import Output, step
import lightning.pytorch as pl
from satpy import Scene
from h5py import File
import os
import numpy as np
import io


class Sat2RadPreprocessor:
    def __init__(self, base_path: str = "./data"):
        self.base_path = base_path

    def load_files(self) -> tuple[list, list]:
        self.radar_files = os.listdir(os.path.join(self.base_path, "radar"))
        self.satellite_files = os.listdir(os.path.join(self.base_path, "satellite"))

    def preprocess_radar_file(self, path):
        path = os.path.join(self.base_path, "radar", path)
        radarFile = File(path)
        return np.array(radarFile["reflectivity"])

    def preprocess_satellite_file(self, path):
        channels = ["VIS006", "VIS008", "IR_120", "IR_134"]
        path = os.path.join(self.base_path, "satellite", path)
        scn = Scene(reader="seviri_l1b_native", filenames=[path])
        scn.load(channels)
        loaded_channels = [scn[x].values for x in channels]
        return np.array(loaded_channels)

    def preprocess(self):
        print("loading files")
        self.load_files()
        print("starting preprocessing")

        for rf in self.radar_files:
            rRes = self.preprocess_radar_file(rf)
            np.save(
                os.path.join(
                    self.base_path,
                    "preprocessed",
                    "radar",
                    rf.replace(".h5", ""),
                ),
                rRes,
            )

        for sf in self.satellite_files:
            sRes = self.preprocess_satellite_file(sf)
            np.save(
                os.path.join(
                    self.base_path,
                    "preprocessed",
                    "satellite",
                    sf.replace(".nat", ""),
                ),
                sRes,
            )


@step
def preprocessor() -> None:
    pr = Sat2RadPreprocessor(base_path="../../../../../data")
    pr.preprocess()


if __name__ == "__main__":
    pr = Sat2RadPreprocessor(base_path="../../../../../data")

    pr.preprocess()
