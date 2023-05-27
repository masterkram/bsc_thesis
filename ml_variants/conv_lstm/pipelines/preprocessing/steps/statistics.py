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

PATH_TO_DATA = "../../../../data"
RADAR_PARAMETER = "reflectivity"


@step
def get_statistics(
    satellite_data: list, radar_data: list
) -> Output(satellite_stats=dict, radar_stats=dict):
    max_radar_value = 0
    for radar_image in radar_data:
        path = os.path.join(PATH_TO_DATA, "radar", radar_image)
        radarFile = File(path)
        radar = np.array(radarFile[RADAR_PARAMETER])
        current_image_max = np.max(radar)
        if current_image_max > max_radar_value:
            max_radar_value = current_image_max

    return (
        {},
        {"max": max_radar_value},
    )
