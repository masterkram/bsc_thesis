from zenml import step
import os
import datetime
from lib.service.BucketService import BucketService
from functools import cmp_to_key
from util.parse_time import parseTime
import zipfile
import cv2
from typing import List

import warnings
from tqdm import tqdm
from util.log_utils import write_log

import typed_settings as ts
from Settings import RadarSettings, SatelliteSettings, DownloadSettings
import os


radarSettings = ts.load(RadarSettings, "radar", config_files=["config.toml"])
satSettings = ts.load(SatelliteSettings, "satellite", config_files=["config.toml"])
downloadSettings = ts.load(DownloadSettings, "downloads", config_files=["config.toml"])


@step
def download_data() -> None:
    """
    First step in the pipeline, downloads the data.
    """
    # check if necessary
    if not downloadSettings.skip:
        bucketService = BucketService()
        bucketService.getFiles()
        time_span = (downloadSettings.range.start, downloadSettings.range.end)
        bucketService.downloadFilesInRange(time_span=time_span)
        unzip()
    else:
        write_log("downloads skipped")


def unzip() -> None:
    path = satSettings.folder.original_path
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
        if not file.endswith(satSettings.folder.file_ext):
            print("removing")
            os.remove(f"{path}/{file}")
