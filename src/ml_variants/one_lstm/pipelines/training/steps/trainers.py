import sys
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from axial_attention import AxialAttention, AxialPositionalEmbedding

sys.path.append("../../")
sys.path.append("../../../")

from zenml import step
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch import optim
import mlflow
from mlflow import MlflowClient
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from layers.ConvLSTM import ConvLSTM
import lightning.pytorch as pl

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import typed_settings as ts
from Settings import ModelSettings, MlFlowSettings
from util.log_utils import write_log, log_mlflow
import numpy as np
from datetime import datetime
from util.evaluate_regression import RegressionMetrics

settings = ts.load(ModelSettings, "model", ["config.toml"])
mlflowConfig = ts.load(MlFlowSettings, "mlflow", ["config.toml"])


mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=mlflowConfig.experiment_name
)


def weight_function(target):
    result = target * 255
    result = target[target < 2] = 1
    mask = torch.logical_and(target < 5, target >= 2)
    result = target[mask] = 2
    mask = torch.logical_and(target < 10, target >= 5)
    result = target[mask] = 5
    mask = torch.logical_and(target < 30, target >= 10)
    result = target[mask] = 10
    result = target[target >= 30] = 30
    return result


SAVE_FOLDER = "../../../../../logs/"


def save_viz(image: np.ndarray, gt: np.ndarray, sat: np.ndarray, idx):
    fig, axes = plt.subplots(2, 5)
    for i, a in enumerate(axes[0]):
        a.imshow(sat[i][0])
    # for i, a in enumerate(axes[1]):
    #     a.set_title("timestamp")
    #     a.imshow(radar[i])
    axes[-1, 0].set_title("ground truth")
    axes[-1, 0].imshow(gt)
    axes[-1, 1].axis("off")
    axes[-1, 2].axis("off")
    axes[-1, 3].axis("off")
    axes[-1, -1].set_title("prediction")
    axes[-1, -1].imshow(image)

    save_path = f"{SAVE_FOLDER}experiment-{idx}.png"
    fig.savefig(save_path, dpi=300)


def balanced_mae(output, target):
    return torch.mean(torch.abs((target - output) * weight_function(target)))


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.temporal_encoder = ConvLSTM(
            input_dim=11,
            hidden_dim=64,
            kernel_size=3,
            num_layers=3,
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(5, 5), padding="same"),
            nn.Conv2d(32, 16, kernel_size=(5, 5), padding="same"),
            nn.Conv2d(16, 1, kernel_size=(5, 5), padding="same"),
        )

        self.loss_fn = nn.MSELoss()
        self.metrics = RegressionMetrics()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.metrics.eval(self, y_hat, y, "train", self.device)

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.metrics.eval(self, y_hat, y, "train", self.device)

    def test_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.metrics.eval(self, y_hat, y, "train", self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, _ = batch

        # x = x.contiguous().view(-1, 11, 256, 256)
        # x = self.cnn_encoder(x).view(2, 5, 64, 256, 256)

        temporal_encoded, state = self.temporal_encoder(x)

        y_hat = self.head(temporal_encoded[:, -1, :, :, :])

        return y_hat


@step(
    enable_cache=False,
    experiment_tracker=mlflowConfig.experiment_tracker,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader, val_dataloader: DataLoader) -> nn.Module:
    write_log(f"training {settings.name}")
    model = Sat2Rad()

    trainer = pl.Trainer(max_epochs=settings.max_epochs)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return model
