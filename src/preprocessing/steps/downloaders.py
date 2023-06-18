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
        bucketService = BucketService(data_folder_path=downloadSettings.directory)
        files = bucketService.getFiles()
        write_log(f"downloading {len(files)} files :checkered_flag:")
        time_span = (downloadSettings.range.start, downloadSettings.range.end)
        if downloadSettings.range.all:
            write_log("downloading all files :folded_hands:")
            bucketService.downloadAllFiles(loadBar=True)
        else:
            write_log(f"downloading file in range {time_span} :four_o’clock:")
            bucketService.downloadFilesInRange(time_span=time_span, loadBar=True)
        unzip()
    else:
        write_log("downloads skipped :floppy_disk:")


def unzip() -> None:
    write_log("extracting zip files :zipper-mouth_face:")
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
            write_log(f"removing unused file: {file} :broom:")
            os.remove(f"{path}/{file}")
