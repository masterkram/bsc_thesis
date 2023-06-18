import typed_settings as ts
from typing import Dict, List
import datetime


@ts.settings
class Shape:
    height: int
    width: int
    channels: int
    sequence_length: int


@ts.settings
class ConvLSTMSettings:
    kernel_size: List
    layers: int
    filters: int


@ts.settings
class DatasetSplit:
    training: float
    validation: float
    testing: float


@ts.settings
class ModelSettings:
    name: str
    batch_size: int
    max_epochs: int
    loss: str
    splits: DatasetSplit
    input_size: Shape
    output_size: Shape
    encoder: ConvLSTMSettings


@ts.settings
class MlFlowSettings:
    experiment_name: str
    experiment_tracker: str


@ts.settings
class VisualizeSettings:
    output_dir: str
