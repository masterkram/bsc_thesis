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
from util.evaluate_classification import ClassifierMetrics
from lib.loss.dice import DiceLoss

settings = ts.load(ModelSettings, "model", ["config.toml"])
mlflowConfig = ts.load(MlFlowSettings, "mlflow", ["config.toml"])
SAVE_FOLDER = "../../../../../logs/"


def save_viz(image: np.ndarray, gt: np.ndarray, sat: np.ndarray, idx):
    fig, axes = plt.subplots(2, 5)
    for i, a in enumerate(axes[0]):
        a.imshow(sat[i][0])

    axes[-1, 0].set_title("ground truth")
    axes[-1, 0].imshow(gt)
    axes[-1, 1].axis("off")
    axes[-1, 2].axis("off")
    axes[-1, 3].axis("off")
    axes[-1, -1].set_title("prediction")
    axes[-1, -1].imshow(image)

    fig.tight_layout()
    save_path = f"{SAVE_FOLDER}experiment-{idx}.png"
    fig.savefig(save_path)


mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=mlflowConfig.experiment_name
)


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.initial_encode = nn.Sequential(
            nn.Conv2d(12, 120, (3, 3), 1, padding=1),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(120, 256, (3, 3), 1, padding=1),
            nn.Conv2d(256, 256, (3, 3), 1, padding=1),
            nn.Conv2d(256, 256, (3, 3), 1, padding=1),
            nn.MaxPool2d((2, 2), 2),
        )

        self.temporal_encoder = ConvLSTM(
            input_dim=256,
            hidden_dim=384,
            kernel_size=3,
            num_layers=3,
        )

        self.spatial_aggregator = nn.Sequential(
            nn.Conv2d(384, 64, (3, 3), 1, "same"),
            nn.Tanh(),
            nn.Conv2d(64, 32, (5, 5), 1, "same"),
            nn.Tanh(),
            nn.Conv2d(32, 8, (3, 3), 1, "same"),
        )
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

        self.loss_fn = DiceLoss(weight=weigths)
        self.metrics = ClassifierMetrics(settings.training.metrics)

    def training_step(self, batch, batch_idx):
        x, y = batch

        h = torch.nn.functional.one_hot(y, 8).view(1, 8, 64, 64)

        y = torch.squeeze(y, 1)
        print(h.shape, "h")
        y_hat = self(batch)

        loss = self.loss_fn(y_hat, h)

        test = torch.unique(torch.argmax(y_hat, 1))
        if torch.sum(test) != 0:
            write_log(f"not 0 :smile: == {test}")

        if batch_idx % 50 == 0:
            write_log("saving visualization for batch")
            save_viz(
                torch.argmax(y_hat, 1).view(64, 64).cpu().detach().numpy(),
                y.view(64, 64).cpu().detach().numpy(),
                x.view(5, 12, 256, 256).cpu().detach().numpy(),
                batch_idx,
            )

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        self.metrics.evaluate_classification(self, y_hat, y, "train", self.device)

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        h = torch.nn.functional.one_hot(y, 8).view(1, 8, 64, 64)
        print(h.shape, "h")

        y = torch.squeeze(y, 1)

        y_hat = self(batch)
        loss = self.loss_fn(y_hat, h)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.metrics.evaluate_classification(self, y_hat, y, "val", self.device)

    def test_step(self, batch, batch_idx):
        _, y = batch
        h = torch.nn.functional.one_hot(y, 8).view(1, 8, 64, 64)

        y = torch.squeeze(y, 1)

        y_hat = self(batch)
        loss = self.loss_fn(y_hat, h)
        self.log("test_loss", loss, on_epoch=True)
        self.metrics.evaluate_classification(self, y_hat, y, "test", self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.argmax(self(batch), dim=1)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, _ = batch

        x = x.view(-1, settings.input_size.channels, 256, 256)
        x = self.initial_encode(x)
        x = x.view(1, settings.input_size.sequence_length, 256, 64, 64)

        x, state = self.temporal_encoder(x)

        # return self.spatial_aggregator(state[-1][0])
        print(x.shape)
        return self.spatial_aggregator(x[:, -1, :, :, :].view(1, -1, 64, 64))


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
