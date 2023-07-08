from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from zenml.steps import Output
from zenml import step
from satpy import Scene
import numpy as np
import sys
import typed_settings as ts
from typing import Dict

sys.path.append("../")

sys.path.append("../../../../")
sys.path.append("../../../../lib/data_loaders")

from lib.data_loaders.Sat2RadModule import Sat2RadDataModule
from lib.data_loaders.DatasetType import DatasetType
from Settings import ModelSettings

settings = ts.load(ModelSettings, "model", ["config.toml"])

PATH_TO_DATA = "../../../../../data"


@step(enable_cache=False)
def importer_sat2rad() -> (
    Output(
        train_dataloader=DataLoader,
        val_dataloader=DataLoader,
        test_dataloader=DataLoader,
        predict_dataloader=DataLoader,
        file_invite_list=Dict,
    )
):
    data_module = Sat2RadDataModule(
        data_dir=PATH_TO_DATA,
        batch_size=1,
        sequence_len_satellite=settings.input_size.sequence_length,
        sequence_len_radar=settings.output_size.sequence_length,
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
        dataset_type=DatasetType.ClassSlidingWindow,
        regression=False,
    )
    data_module.prepare_data()
    data_module.setup(None)
    file_invite_list = data_module.get_files()
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
        data_module.predict_dataloader(),
        file_invite_list,
    )
