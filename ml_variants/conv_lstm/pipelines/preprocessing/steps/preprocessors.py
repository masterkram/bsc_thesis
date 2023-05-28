from zenml.steps import Output, step
from satpy import Scene
from h5py import File
import os
import numpy as np
from pyresample.geometry import create_area_def
import datetime
from BucketService import BucketService
from functools import cmp_to_key
from parse_time import parseTime
import zipfile
import cv2

import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


READER = "seviri_l1b_native"
PROJECTION = "+proj=merc +lat_0=52.5 +lon_0=5.5 +ellps=WGS84"
SAT_CHANNELS = ["VIS006", "VIS008", "IR_120", "IR_134"]
RADAR_PARAMETER = "reflectivity"
custom_area = create_area_def(
    "my_area",
    PROJECTION,
    width=134,
    height=166,
    area_extent=[0, 50, 10, 55],
    units="degrees",
)
PATH_TO_DATA = "../../../../data"


def preprocess_radar_file(path: str, stats: dict):
    rescale_ratio = 1 / stats["max"] if stats["max"] is not None else 1 / 255

    path = os.path.join(PATH_TO_DATA, "radar", path)
    radarFile = File(path)
    radar = np.array(radarFile[RADAR_PARAMETER])

    # cut border mask with value 255
    radar[radar >= 255] = 0

    # normalize pixels between 0-1
    radar = radar * rescale_ratio
    resizeRadar = cv2.resize(radar, (134, 166))
    return resizeRadar


def preprocess_satellite_file(path):
    path = os.path.join(PATH_TO_DATA, "satellite", path)
    scn = Scene(reader=READER, filenames=[path])
    scn.load(SAT_CHANNELS)
    local_scn = scn.resample(custom_area, resampler="nearest")
    loaded_channels = [local_scn[x].values for x in SAT_CHANNELS]
    return np.array(loaded_channels)


@step
def download_data() -> None:
    """
    First step in the pipeline, downloads the data.
    """
    # check if necessary
    flag = False
    if flag:
        bucketService = BucketService()
        bucketService.getFiles()
        time_span = (
            datetime.datetime(2023, 4, 21, 0),
            datetime.datetime(2023, 4, 21, 23),
        )
        bucketService.downloadFilesInRange(time_span=time_span)
        unzip()
    else:
        print("===== skipping downloads ======")


@step
def load_data() -> Output(satellite_images=list, radar_images=list):
    """
    Second step in the pipeline gets available files in the respective folders.
    Returns lists of files ordered by time.
    """
    radar_images = os.listdir(f"{PATH_TO_DATA}/radar")
    satellite_images = os.listdir(f"{PATH_TO_DATA}/satellite")

    radar_images = order_based_on_file_timestamp(radar_images)
    satellite_images = order_based_on_file_timestamp(satellite_images)

    return satellite_images, radar_images


def compare_files(file1: str, file2: str) -> int:
    date1, date2 = parseTime(file1), parseTime(file2)
    if date1 > date2:
        return 1
    elif date1 < date2:
        return -1

    return 0


def order_based_on_file_timestamp(files: list) -> list:
    return sorted(files, key=cmp_to_key(compare_files))


def unzip() -> None:
    path = f"{PATH_TO_DATA}/satellite"
    zips = os.listdir(path)

    for file in zips:
        if file.endswith(".zip"):
            zip_path = f"{path}/{file}"

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path)

            if os.path.exists(zip_path):
                os.remove(zip_path)

            zip_ref.close()

    # clean other files
    files = os.listdir(path)
    for file in files:
        if not file.endswith(".nat"):
            print("removing")
            os.remove(f"{path}/{file}")


@step
def preprocess_satellite(filenames: list[str]) -> None:
    """
    Preprocessing of satellite data.
    """
    for file in filenames:
        result = preprocess_satellite_file(file)
        np.save(
            os.path.join(
                PATH_TO_DATA,
                "preprocessed",
                "satellite",
                file.replace(".nat", ""),
            ),
            result,
        )


@step
def preprocess_radar(filenames: list[str], stats: dict) -> None:
    for file in filenames:
        result = preprocess_radar_file(file, stats)
        np.save(
            os.path.join(
                PATH_TO_DATA,
                "preprocessed",
                "radar",
                file.replace(".h5", ""),
            ),
            result,
        )


@step
def visualize_satellite_data(filenames: list) -> np.ndarray:
    return np.ones((400, 400))


@step
def visualize_radar_data(filenames: list) -> np.ndarray:
    return np.ones((400, 400))
