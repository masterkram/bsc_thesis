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
from util.evaluate_classification import ClassifierMetrics
import torchmetrics
from layers.Unet3d import UNet3D

settings = ts.load(ModelSettings, "model", ["config.toml"])
mlflowConfig = ts.load(MlFlowSettings, "mlflow", ["config.toml"])

SAVE_FOLDER = "../../../../../logs/"


mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=mlflowConfig.experiment_name
)


def save_viz(pred, gt, i):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(pred)
    axes[1].imshow(gt)
    fig.savefig(f"{SAVE_FOLDER}testunet-weights-{i}.png", dpi=300)


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet3D(settings.input_size.channels, settings.classes, num_groups=1)
        self.classifer = nn.Conv3d(
            settings.input_size.sequence_length, settings.output_size.channels, 3, 1, 1
        )
        weights = torch.tensor(settings.training.class_weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

        self.metrics = ClassifierMetrics(
            settings.training.metrics, num_classes=settings.classes
        )
        print(self.device)

        self.softmax = torch.nn.Softmax2d()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(batch)

        # loss
        y = torch.squeeze(y, 1)

        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        # log metrics:
        # self.accuracy(y_hat, y)
        # self.log("train_acc", self.accuracy, on_epoch=True, prog_bar=True)
        self.metrics.evaluate_classification(self, y_hat, y, "train", self.device)

        test = torch.unique(torch.argmax(y_hat, 1))
        if torch.sum(test) != 0:
            write_log(f"not 0 :smile: == {test}")

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch

        y_hat = self(batch)
        y = torch.squeeze(y, 1)

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.metrics.evaluate_classification(self, y_hat, y, "val", self.device)

    def test_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        y = torch.squeeze(y, 1)

        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        self.metrics.evaluate_classification(self, y_hat, y, "test", self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self(batch)

        return y_hat

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        x = x.reshape(-1, 12, 8, 256, 256)
        x = self.unet(x)
        x = x.reshape(-1, 8, 8, 256, 256)
        x = torch.squeeze(self.classifer(x), 1)

        return x


@step(
    enable_cache=False,
    experiment_tracker=mlflowConfig.experiment_tracker,
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader, val_dataloader: DataLoader) -> nn.Module:
    write_log(f"training {settings.name}")
    model = Sat2Rad()

    trainer = pl.Trainer(max_epochs=40)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return model
