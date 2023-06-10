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


def save_viz(
    image: np.ndarray, active_experiment_name: str, pred: bool = True, batch_ix=0
):
    image = np.reshape(image, (256, 256))
    plt.imshow(image)

    # save the plot
    save_type = "pred" if pred else "gt"
    save_path = (
        f"{SAVE_FOLDER}experiment-{active_experiment_name}-{save_type}-{batch_ix}.png"
    )
    plt.savefig(save_path, dpi=300)


def balanced_mae(output, target):
    return torch.mean(torch.abs((target - output) * weight_function(target)))


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.temporal_encoder = ConvLSTM(
            input_size=(settings.input_size.height, settings.input_size.width),
            input_dim=settings.input_size.channels,
            hidden_dim=[
                settings.encoder.filters for _ in range(settings.encoder.layers)
            ],
            kernel_size=tuple(settings.encoder.kernel_size),
            num_layers=settings.encoder.layers,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.relu,
        )

        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding="same"),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding="same"),
        )

        self.loss_fn = nn.L1Loss()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y[0])

        # sanity check visualisation
        # save_viz(y_hat.detach().cpu().numpy(), "booya", True, batch_idx)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y[0])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y[0])
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, _ = batch
        hidden_state = self.temporal_encoder.get_init_states(
            batch_size=1, device=self.device
        )
        temporal_encoded, state = self.temporal_encoder(x, hidden_state)

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

    trainer = pl.Trainer(
        max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
    )

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return model
