import sys
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from axial_attention import AxialAttention, AxialPositionalEmbedding

sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

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
import torchmetrics
from lib.loss.glou import giou_loss
from lib.loss.focal_loss import FocalLoss
from layers.Lstm2 import ConvLSTMBlock
from util.evaluate_regression import RegressionMetrics

settings = ts.load(ModelSettings, "model", ["config.toml"])
mlflowConfig = ts.load(MlFlowSettings, "mlflow", ["config.toml"])
SAVE_FOLDER = "../../../../../logs/"


mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=mlflowConfig.experiment_name
)


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.temporal_encoder = ConvLSTM(
            input_dim=64,
            hidden_dim=64,
            kernel_size=3,
            num_layers=2,
        )

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=34, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=34, out_channels=64, kernel_size=3, stride=2),
        )

        self.position_embedding = AxialPositionalEmbedding(dim=64, shape=(62, 62))

        self.temporal_agg = nn.Sequential(
            *[
                AxialAttention(dim=64, dim_index=1, heads=4, num_dimensions=2)
                for _ in range(1)
            ]
        )

        self.head = nn.Conv2d(
            64, 1, kernel_size=(1, 1), padding="same"
        )  # Reduces to mask

        self.loss_fn = nn.MSELoss()
        self.metrics = RegressionMetrics()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = torch.squeeze(y, 1)

        y_hat = self(batch)
        print(y.shape, y_hat.shape)

        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        self.metrics.eval(self, y_hat, y, "train", self.device)

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y = torch.squeeze(y, 1)

        y_hat = self(batch)
        print(y.shape, y_hat.shape)

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.metrics.eval(self, y_hat, y, "val", self.device)

    def test_step(self, batch, batch_idx):
        _, y = batch
        y = torch.squeeze(y, 1)

        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)
        self.metrics.eval(self, y_hat, y, "test", self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, _ = batch

        x_for_conv = x.view(-1, 11, 256, 256)
        x = self.encode(x_for_conv).view(
            1, settings.input_size.sequence_length, 64, 62, 62
        )

        _, state = self.temporal_encoder(x)

        # # print(state, "this is state")
        # print(len(state[0]), "this is state 0 is h ?")

        embedding = self.position_embedding(state[-1][0])

        xi = self.temporal_agg(embedding)

        y_hat = self.head(xi)

        return torch.squeeze(y_hat, 1)


@step(
    enable_cache=False,
    experiment_tracker=mlflowConfig.experiment_tracker,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader, val_dataloader: DataLoader) -> nn.Module:
    write_log(f"training {settings.name}")
    model = Sat2Rad()

    trainer = pl.Trainer(max_epochs=50)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return model
