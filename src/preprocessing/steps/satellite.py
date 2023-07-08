from zenml import step
from satpy import Scene
import os
import numpy as np
from pyresample.geometry import create_area_def
from typing import List, Dict, Type
import warnings
from tqdm import tqdm
from BindFiles import BindFiles
import typed_settings as ts
from Settings import SatelliteSettings
import sys
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

sys.path.append("../")

from util.log_utils import write_log

warnings.filterwarnings("ignore")

settings = ts.load(SatelliteSettings, "satellite", config_files=["config.toml"])

custom_area = create_area_def(
    "my_area",
    settings.reprojection.projection_string,
    width=settings.output_size.width,
    height=settings.output_size.height,
    area_extent=settings.reprojection.area,
    units="degrees",
)


class SatelliteStatistics:
    def __init__(self, maxes, mins):
        self.maxes = maxes
        self.mins = mins


class MyMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (SatelliteStatistics,)
    ASSOCIATED_ARTIFACT_TYPES = (ArtifactType.DATA,)

    def load(self, data_type: Type[SatelliteStatistics]) -> SatelliteStatistics:
        """Read from artifact store"""
        super().save(data_type)
        with fileio.open(os.path.join(self.uri, "data.npz"), "rb") as f:
            data = np.load(f)
            maxes = data["maxes"]
            mins = data["mins"]

        return SatelliteStatistics(maxes=maxes, mins=mins)

    def save(self, my_obj: SatelliteStatistics) -> None:
        """Write to artifact store"""
        super().load(my_obj)
        with fileio.open(os.path.join(self.uri, "data.npz"), "wb") as f:
            np.savez(f, maxes=my_obj.maxes, mins=my_obj.mins)


def reproject_sat_file(file: str) -> np.ndarray:
    """
    Reprojects a satellite file and saves channels of this file in a numpy file.
    """
    path = os.path.join(settings.folder.original_path, file)
    scn = Scene(reader=settings.reader, filenames=[path])
    scn.load(settings.channels)
    local_scn = scn.resample(custom_area, resampler=settings.reprojection.resampler)
    loaded_channels = [local_scn[x].values for x in settings.channels]
    return np.array(loaded_channels)


def min_max_scale(x, maxes, mins):
    """
    Scale a numpy array between 0 and 1
    """
    return (x - mins) / (maxes - mins)


def get_scaler_function(stats: SatelliteStatistics):
    """
    Return a function that scales an input numpy file.
    """
    pixels = settings.output_size.height * settings.output_size.width

    channels = len(settings.channels)

    mins = np.repeat(stats.mins, pixels)
    mins = mins.reshape(
        channels, settings.output_size.height, settings.output_size.width
    )
    maxes = np.repeat(stats.maxes, pixels)
    maxes = maxes.reshape(
        channels, settings.output_size.height, settings.output_size.width
    )

    def min_max_scale_file(file: str) -> np.ndarray:
        path = os.path.join(
            settings.folder.save_path, file.replace(settings.folder.file_ext, ".npy")
        )
        arr = np.load(path)

        arr = min_max_scale(arr, maxes, mins)
        # confirm scaling
        write_log(
            f"max value in array: {np.max(arr)}, min value in array: {np.min(arr)}"
        )

        return arr

    return min_max_scale_file


def rename(original: str):
    return os.path.join(
        settings.folder.save_path, original.replace(settings.folder.file_ext, "")
    )


@step
def reproject_satellite(filenames: List[str]) -> None:
    if not settings.reprojection.skip:
        satFiles = BindFiles(filenames, rename)
        satFiles.bind(reproject_sat_file, desc="reprojecting satellite")
    else:
        write_log("skipping reprojection of satellite")


@step
def satellite_pixel_normalization(
    filenames: List[str], stats: SatelliteStatistics
) -> None:
    """
    zenml step to normalize pixels.
    """
    satFiles = BindFiles(filenames, rename)
    satFiles.bind(get_scaler_function(stats), desc="normalize pixel values")


@step
def get_satellite_stats(satellite_data: List) -> SatelliteStatistics:
    """
    get statistics of dataset.
    - max per channel
    - min per channel
    """
    channel_length = len(settings.channels)
    maxes = np.zeros((channel_length,))
    mins = np.zeros((channel_length,))

    for sat_file in tqdm(satellite_data, desc="calculating sat statistics"):
        path = os.path.join(
            settings.folder.save_path,
            sat_file.replace(settings.folder.file_ext, ".npy"),
        )
        arr = np.load(path)

        local_maxes = np.amax(arr, axis=(1, 2))
        local_mins = np.amin(arr, axis=(1, 2))

        for idx, value in enumerate(maxes):
            if value < local_maxes[idx]:
                maxes[idx] = local_maxes[idx]

        for idx, value in enumerate(mins):
            if value > local_mins[idx]:
                mins[idx] = local_mins[idx]

    write_log(f"maximum values per channel: {maxes}")
    write_log(f"minimum values per channel: {mins}")
    stats = SatelliteStatistics(maxes, mins)
    return stats


@step()
def visualize_satellite(satellite_files: List[str]) -> np.ndarray:
    """
    Visualize the entire dataset.
    """
    # sample = satellite_files[-1].replace(settings.folder.file_ext, ".npy")
    # sample = os.path.join(settings.folder.save_path, sample)
    # sample = np.load(sample)
    # return sample[0]
    return np.zeros((4, 4))
