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
from torch import nn
from models.unet_parts import OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.layers import CBAM

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


class SmaAt_UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    ):
        super(SmaAt_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(
            self.n_channels, 64, kernels_per_layer=kernels_per_layer
        )
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(
            1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer
        )
        self.up2 = UpDS(
            512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer
        )
        self.up3 = UpDS(
            256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer
        )
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet3D(11, 14, num_groups=1)
        self.classifer = nn.Conv3d(8, 1, 3, 1, 1)
        weights = torch.tensor(
            [
                0.001,
                0.001,
                0.002,
                0.003,
                0.004,
                0.009,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ]
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=14)
        self.softmax = torch.nn.Softmax2d()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(batch)

        # y_hat = torch.argmax(y_hat, 1)
        # loss
        y = torch.squeeze(y, 1)

        print(y.shape, "y shape")
        print(y_hat, "shape y hat")
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        # accuracy
        self.accuracy(y_hat, y)
        self.log("train_acc", self.accuracy, on_epoch=True, prog_bar=True)

        save_viz(
            torch.argmax(y_hat, 1).view(256, 256).cpu().detach().numpy(),
            y.view(256, 256).cpu().detach().numpy(),
            batch_idx,
        )

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

        loss = self.loss_fn(y_hat, y[0])
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self(batch)

        return torch.argmax(y_hat, 1)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        x = x.reshape(-1, 11, 8, 256, 256)
        x = self.unet(x)
        x = x.reshape(-1, 8, 14, 256, 256)
        x = torch.squeeze(self.classifer(x), 1)
        x = self.softmax(x)

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
