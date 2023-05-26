from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from enum import Enum
from zenml.steps import Output, step
import lightning.pytorch as pl
import numpy as np


@step
def visualize(predict_dataloader: DataLoader, model: pl.LightningModule) -> np.ndarray:
    trainer = pl.Trainer()
    result = trainer.predict(model, predict_dataloader)

    return result["predict_dataloader"].detach().numpy()
