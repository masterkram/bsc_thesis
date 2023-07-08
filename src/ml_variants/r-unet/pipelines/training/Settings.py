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
class UnetSettings:
    kernel_size: List
    layers: int
    filters: int


@ts.settings
class TrainingSettings:
    max_epochs: int
    class_weights: List[float]
    metrics: List[str]


@ts.settings
class ModelSettings:
    name: str
    classes: int
    input_size: Shape
    output_size: Shape
    unet: UnetSettings
    training: TrainingSettings


@ts.settings
class MlFlowSettings:
    experiment_name: str
    experiment_tracker: str


@ts.settings
class VisualizeSettings:
    output_dir: str
