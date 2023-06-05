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
    radarFile[radarFile >= settings.pixel_range[1]] = 0
    rescaleRatio = 1 / (settings.pixel_range[1] - settings.pixel_range[0])
    radarFile = radarFile * rescaleRatio
    return radarFile


def resize_file(path: str) -> np.ndarray:
    path = os.path.join(
        settings.folder.save_path, path.replace(settings.folder.file_ext, ".npy")
    )
    radarFile = np.load(path)
    resizeRadar = cv2.resize(
        radarFile, (settings.output_size.width, settings.output_size.height)
    )
    return resizeRadar


def rename(original: str) -> str:
    return os.path.join(
        settings.folder.save_path, original.replace(settings.folder.file_ext, "")
    )


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
    files = BindFiles(filenames, rename)
    files.bind(resize_file)


@step
def visualize_radar(filenames: List[str]) -> np.ndarray:
    sample = filenames[-1].replace(settings.folder.file_ext, ".npy")
    sample = os.path.join(settings.folder.save_path, sample)
    sample = np.load(sample)
    return sample[0]
