from zenml.steps import Output
from zenml import step
from typing import List
from util.parse_time import order_based_on_file_timestamp
import typed_settings as ts
from Settings import RadarSettings, SatelliteSettings
import os

import sys

sys.path.append("../../../../")
sys.path.append("../../../../../")

from util.log_utils import write_log


radarSettings = ts.load(RadarSettings, "radar", config_files=["config.toml"])
satSettings = ts.load(SatelliteSettings, "satellite", config_files=["config.toml"])


@step
def load_data() -> Output(satellite_images=List, radar_images=List):
    """
    Second step in the pipeline gets available files in the respective folders.
    Returns lists of files ordered by time.
    """
    radar_images = os.listdir(radarSettings.folder.original_path)
    satellite_images = os.listdir(satSettings.folder.original_path)

    write_log(f"loaded {len(satellite_images)} satellite images :rocket:")
    write_log(f"loaded {len(radar_images)} radar images :satellite:")

    radar_images = order_based_on_file_timestamp(radar_images)
    satellite_images = order_based_on_file_timestamp(satellite_images)

    return satellite_images, radar_images
