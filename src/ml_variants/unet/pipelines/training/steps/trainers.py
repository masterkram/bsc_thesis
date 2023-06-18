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
        self.unet = UNet3D(12, 8, num_groups=1)
        self.classifer = nn.Conv3d(8, 1, 3, 1, 1)
        weights = torch.tensor(
            [
                0.01081153,
                0.13732371,
                0.13895907,
                0.1416087,
                0.14272867,
                0.14285409,
                0.14285709,
                0.14285714,
            ]
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=8)
        self.softmax = torch.nn.Softmax2d()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(batch)

        # y_hat = torch.argmax(y_hat, 1)
        # loss
        y = torch.squeeze(y, 1)

        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        # accuracy
        self.accuracy(y_hat, y)
        self.log("train_acc", self.accuracy, on_epoch=True, prog_bar=True)

        test = torch.unique(torch.argmax(y_hat, 1))
        if torch.sum(test) != 0:
            write_log(f"not 0 :smile: == {test}")
        # save_viz(
        #     torch.argmax(y_hat, 1).view(256, 256).cpu().detach().numpy(),
        #     y.view(256, 256).cpu().detach().numpy(),
        #     batch_idx,
        # )

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch

        y_hat = self(batch)
        y = torch.squeeze(y, 1)

        print(y.shape, "y shape")
        print(y_hat.shape, "shape y hat")
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        y = torch.squeeze(y, 1)

        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self(batch)

        # return torch.argmax(y_hat, 1)
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

    trainer = pl.Trainer(max_epochs=5)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return model
