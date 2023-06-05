import typed_settings as ts
from typing import Dict, List
import datetime


@ts.settings
class Dimension:
    height: int
    width: int


@ts.settings
class ReprojectionSettings:
    skip: bool
    projection_string: str
    area: List
    resampler: str


@ts.settings
class FolderSettings:
    original_path: str
    save_path: str
    file_ext: str


@ts.settings
class SatelliteSettings:
    reader: str
    channels: List
    folder: FolderSettings
    output_size: Dimension
    reprojection: ReprojectionSettings


@ts.settings
class RadarSettings:
    parameter: str
    pixel_range: List
    folder: FolderSettings
    output_size: Dimension


@ts.settings
class DateRange:
    all: bool
    start: datetime.datetime
    end: datetime.datetime


@ts.settings
class DownloadSettings:
    skip: bool
    endpoint: str
    region: str
    bucket_name: str
    digital_ocean: bool
    range: DateRange
    directory: str
