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

settings = ts.load(ModelSettings, "model", ["config.toml"])
mlflowConfig = ts.load(MlFlowSettings, "mlflow", ["config.toml"])
SAVE_FOLDER = "../../../../../logs/"


def save_viz(pred, gt, i):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(pred)
    axes[1].imshow(gt)
    fig.savefig(f"{SAVE_FOLDER}testconvclass-{i}.png", dpi=300)


mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=mlflowConfig.experiment_name
)


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.temporal_encoder = ConvLSTM(
            input_size=(256, 256),
            input_dim=12,
            hidden_dim=[64, 64],
            kernel_size=(3, 3),
            num_layers=2,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )
        # self.temporal_encoder = ConvLSTMBlock(12, 64)

        self.spatial_aggregator = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), 1, "same"),
            nn.ReLU(),
            nn.Conv2d(32, 16, (5, 5), 1, "same"),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 3), 1, "same"),
        )
        # self.loss_fn = FocalLoss([0.25 for _ in range(8)], gamma=2)
        weigths = torch.tensor(
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

        self.loss_fn = nn.CrossEntropyLoss(weight=weigths, label_smoothing=0.1)
        self.iou = torchmetrics.JaccardIndex("multiclass", num_classes=8)
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=8)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = torch.squeeze(y, 1)
        y_hat = self(batch)

        loss = self.loss_fn(y_hat, y)

        test = torch.unique(torch.argmax(y_hat, 1))
        if torch.sum(test) != 0:
            write_log(f"not 0 :smile: == {test}")

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        self.accuracy(y_hat, y)  # compute metrics
        self.log("train_acc_step", self.accuracy, prog_bar=True)  # log metric object

        # save_viz(
        #     torch.argmax(y_hat, 1).view(256, 256).cpu().detach().numpy(),
        #     y.view(256, 256).cpu().detach().numpy(),
        #     batch_idx,
        # )

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y = torch.squeeze(y, 1)

        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.accuracy(y_hat, y)  # compute metrics
        self.log("val_acc_step", self.accuracy, prog_bar=True)  # log metric object

    def test_step(self, batch, batch_idx):
        _, y = batch
        y = torch.squeeze(y, 1)

        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

        accuracy = self.accuracy(y_hat, y)  # compute metrics
        self.log("test_accuracy", accuracy)  # log metric object

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # return torch.argmax(self(batch), dim=1)
        return self(batch)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, _ = batch
        hidden_state = self.temporal_encoder.get_init_states(
            batch_size=1, device=self.device
        )
        temporal_encoded, _ = self.temporal_encoder(x, hidden_state)

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

    trainer = pl.Trainer(max_epochs=150)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return model
