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
import torchmetrics

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


def balanced_mae(output, target):
    return torch.mean(torch.abs((target - output) * weight_function(target)))


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.temporal_encoder = ConvLSTM(
            input_size=(256, 256),
            input_dim=11,
            hidden_dim=[64],
            kernel_size=(3, 3),
            num_layers=1,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )

        self.spatial_aggregator = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), 1, "same"),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3, 3), 1, "same"),
            nn.ReLU(),
            nn.Conv2d(16, 14, (3, 3), 1, "same"),
            nn.Softmax2d(),
        )
        weights = torch.tensor(
            [
                0.0001,
                0.001,
                0.002,
                0.003,
                0.004,
                0.009,
                0.01,
                0.01,
                0.04,
                0.05,
                0.12,
                0.15,
                0.2,
                0.4009,
            ]
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.iou = torchmetrics.JaccardIndex("multiclass", num_classes=14)
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=14)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(batch)
        print("size check y = ", y.size(), "y_hat =", y_hat.size())

        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.iou(y_hat, y)  # compute metrics
        self.log("train_iou_step", self.iou)  # log metric object

        self.accuracy(y_hat, y)  # compute metrics
        self.log("train_acc_step", self.accuracy)  # log metric object

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        print("size check y = ", y.size(), "y_hat =", y_hat.size())
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        self.iou(y_hat, y)  # compute metrics
        self.log("val_iou_step", self.iou)  # log metric object
        self.accuracy(y_hat, y)  # compute metrics
        self.log("train_acc_step", self.accuracy)  # log metric object

    def test_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)
        self.iou(y_hat, y)
        self.log("test_iou_step", self.iou)
        self.accuracy(y_hat, y)  # compute metrics
        self.log("train_acc_step", self.accuracy)  # log metric object

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.argmax(self(batch), 1)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, _ = batch
        hidden_state = self.temporal_encoder.get_init_states(
            batch_size=1, device=self.device
        )
        temporal_encoded, state = self.temporal_encoder(x, hidden_state)

        y_hat = self.spatial_aggregator(temporal_encoded[:, -1, :, :, :])

        return y_hat


@step(
    enable_cache=False,
    experiment_tracker=mlflowConfig.experiment_tracker,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader, val_dataloader: DataLoader) -> nn.Module:
    write_log(f"training {settings.name}")
    model = Sat2Rad()

    trainer = pl.Trainer(max_epochs=5)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return model
