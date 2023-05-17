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
from pyresample import geometry
import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config
import datetime

netherlands_extent = geometry.AreaDefinition.from_extent(
    "netherlands",
    projection="+proj=geos h=35785831.0",
    area_extent=(-569000.0, -5569000.0, 9569000.0, 9669000.0),
    shape=(250, 250),
    units="m",
)


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
        local_scn = scn.resample(netherlands_extent)
        # scn.crop(ll_bbox=(-105.0, 40.0, -95.0, 50.0))
        loaded_channels = [local_scn[x].values for x in channels]
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


from_bucket = True
bucket_url = "https://ams3.digitaloceanspaces.com"
bucket_region = 'ams3'
time_span = (datetime.datetime(2023, 4, 21, 0), datetime.datetime(2023, 4, 21, 12))

@step
def download_data() -> None:
    if from_bucket == True:
        session = boto3.session.Session()
        client = session.client(
            "s3",
            endpoint_url=bucket_url,
            region_name=bucket_region,
            config=Config(
                s3={"addressing_style": "virtual"}, signature_version=UNSIGNED
            ),
        )

        for


@step
def preprocessor() -> None:
    pr = Sat2RadPreprocessor(base_path="../../../../data")
    pr.preprocess()


@step
def load_data() -> tuple[list, list]:
    radar_images = []
    satellite_images = []

    return satellite_images, radar_images


@step
def reproject_satellite_images() -> None:
    pass


if __name__ == "__main__":
    pr = Sat2RadPreprocessor(base_path="../../../../data/")

    pr.preprocess()
