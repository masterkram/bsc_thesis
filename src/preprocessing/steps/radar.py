from zenml import step
from typing import List
from tqdm import tqdm
import numpy as np
import typed_settings as ts
from Settings import RadarSettings
from BindFiles import BindFiles
import os
from h5py import File
import cv2
import sys

sys.path.append("../../")
sys.path.append("../")

from util.log_utils import write_log

settings = ts.load(RadarSettings, "radar", config_files=["config.toml"])


def to_numpy(path: str) -> np.ndarray:
    path = os.path.join(settings.folder.original_path, path)
    radarFile = File(path)
    radar = np.array(radarFile[settings.parameter])
    return radar


def normalize_pixels_file(path: str) -> np.ndarray:
    path = os.path.join(
        settings.folder.save_path, path.replace(settings.folder.file_ext, ".npy")
    )
    radarFile = np.load(path)

    # radarFile[radarFile >= settings.pixel_range[1]] = 0
    # rescaleRatio = 1 / (settings.pixel_range[1] - settings.pixel_range[0])
    # radarFile = radarFile * rescaleRatio
    maxes = (settings.pixel_range[1] * 0.5) - 32
    mins = (settings.pixel_range[0] * 0.5) - 32
    return min_max_scale(radarFile, maxes, mins)


def binify(array):
    class_counter = 0
    array[array <= 0] = class_counter
    class_counter += 1
    for i in range(0, 60, 10):
        array[(array > i) & (array <= i + 10)] = class_counter
        class_counter += 1
    array[array > 60] = class_counter
    return array


def bin_file(path: str) -> np.ndarray:
    path = os.path.join(
        settings.folder.save_path, path.replace(settings.folder.file_ext, ".npy")
    )
    radarFile = np.load(path)

    # create bins
    radarFile = binify(radarFile)
    write_log(f"classes of bins {np.unique(radarFile)}")

    return radarFile


def resize_file(path: str) -> np.ndarray:
    path1 = os.path.join(
        settings.folder.save_path, path.replace(settings.folder.file_ext, ".npy")
    )
    path2 = os.path.join(
        settings.save_path_bins, path.replace(settings.folder.file_ext, ".npy")
    )

    radarFile = np.load(path1)
    radarFileBinned = np.load(path2)

    resizeRadar = cv2.resize(
        radarFile, (settings.output_size.width, settings.output_size.height)
    )
    resizeRadarBinned = cv2.resize(
        radarFileBinned, (settings.output_size.width, settings.output_size.height)
    )
    return resizeRadar, resizeRadarBinned


def rename(original: str) -> str:
    return os.path.join(
        settings.folder.save_path, original.replace(settings.folder.file_ext, "")
    )


def renameBins(original: str) -> str:
    return os.path.join(
        settings.save_path_bins, original.replace(settings.folder.file_ext, "")
    )


def dbz(path: str) -> np.ndarray:
    path = os.path.join(
        settings.folder.save_path, path.replace(settings.folder.file_ext, ".npy")
    )
    array = np.load(path)
    # clear mask
    array[array >= settings.pixel_range[1]] = 0
    # convert to dbz
    return (array * 0.5) - 32


def min_max_scale(x, maxes, mins):
    """
    Scale a numpy array between 0 and 1
    """
    return (x - mins) / (maxes - mins)


@step
def create_bins(filenames: List[str]) -> None:
    # files = BindFiles(filenames, renameBins)
    # files.bind(bin_file, "saving the file as binned")
    for file in filenames:
        binned = bin_file(file)
        np.save(renameBins(file), binned)


@step
def to_dbz(filenames: List[str]) -> None:
    files = BindFiles(filenames, rename)
    files.bind(dbz, "converting to dbz")


@step
def convert_radar_to_numpy(filenames: List[str]) -> None:
    files = BindFiles(filenames, rename)
    files.bind(to_numpy, "converting radar files to numpy")


@step
def normalize_radar_pixels(filenames: List[str]) -> None:
    files = BindFiles(filenames, rename)
    files.bind(normalize_pixels_file, "normalizing radar pixels")


@step
def resize_radar_files(filenames: List[str]) -> None:
    write_log("skipping resize")
    # for file in filenames:
    #     normal, binned = resize_file(file)
    #     np.save(rename(file), normal)
    #     np.save(renameBins(file), binned)


@step
def visualize_radar(filenames: List[str]) -> np.ndarray:
    sample = filenames[-1].replace(settings.folder.file_ext, ".npy")
    sample = os.path.join(settings.folder.save_path, sample)
    sample = np.load(sample)
    return sample[0]
