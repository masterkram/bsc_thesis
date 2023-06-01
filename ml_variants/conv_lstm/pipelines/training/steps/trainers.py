import sys
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

sys.path.insert(1, "/Users/mark/Projects/bsc_thesis/ml_variants/conv_lstm")

from zenml.steps import step
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import mlflow
from mlflow import MlflowClient
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
from layers.ConvLSTM import ConvLSTM
import lightning.pytorch as pl

# from utils import log_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


mlflow_settings = MLFlowExperimentTrackerSettings(experiment_name="sat2rad_conv_lstm")


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(2, 3), stride=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(130, 161), stride=1),
        )

    def forward(self, batch):
        result = self.cnn_decoder(batch)
        return result


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=11, out_dim=64):
        super(ConvEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=16,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=self.out_dim,
                kernel_size=(3, 3),
                padding="same",
            ),
        )
        # self.device  = 'cpu'

    # def cuda(self, device='cuda'):
    #     super(ConvEncoder, self).cuda(device)
    #     self.device = device

    def forward(self, input):
        output = self.cnn_encoder(input)
        return output


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
        self.save_hyperparameters()

        self.radar_sequence_length = 12
        self.satellite_channels = 11
        self.input_image_size = (300, 300)
        self.output_image_size = (166, 134)

        self.model_img_dim = 300
        hidden_img = (self.model_img_dim, self.model_img_dim)
        hidden_dim = [64, 64, 64]
        layers = len(hidden_dim)
        channels = 64
        kernel_size = (3, 3)

        self.encoder = ConvLSTM(
            input_size=hidden_img,
            input_dim=channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=layers,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )

        self.decoder = ConvLSTM(
            input_size=hidden_img,
            input_dim=channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=layers,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )

        self.cnn_encoder = ConvEncoder(
            in_channels=self.satellite_channels, out_dim=channels
        )
        self.cnn_decoder = ConvDecoder()

        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(1, 12, 1, 166, 134)
        # y passed => teacher forcing
        # if True:
        #     cnn_encoder_out = self.cnn_encoder(y.contiguous().view(-1, 1, 166, 134))
        #     print(cnn_encoder_out.size())
        #     nextf_enc = cnn_encoder_out.view(1, 12, 64, 166, 134)

        prev_sequence_raw = x.view(
            -1,
            self.satellite_channels,
            self.input_image_size[0],
            self.input_image_size[1],
        )
        prev_sequence_encoded = self.cnn_encoder(prev_sequence_raw).view(
            1, 5, 64, self.model_img_dim, self.model_img_dim
        )

        state = self.encode_temporal(prev_sequence_encoded)

        decoder_output_list = self.decode_temporal(state=state)

        final_decoder_out = torch.cat(decoder_output_list, 1)

        decoder_out_rs = final_decoder_out.view(
            -1, 64, self.model_img_dim, self.model_img_dim
        )
        result = self.cnn_decoder(decoder_out_rs)

        y_hat = result.view(1, 12, 166, 134)

        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss
        # acc = self.calculate_accuracy(yhat, y)

        # self.log("train_loss", loss, on_epoch=True)
        # self.log("acc", acc, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def encode_temporal(self, x: torch.Tensor):
        hidden = self.encoder.get_init_states(batch_size=1, device=self.device)
        _, encoder_state = self.encoder(x, hidden)
        return encoder_state

    def decode_temporal(self, state: torch.Tensor, y=None):
        decoder_output_list = []

        initial_input = torch.ones(
            (1, 1, 64, self.model_img_dim, self.model_img_dim)
        ).to(self.device)

        # call decoder in succession
        for i in range(self.radar_sequence_length):
            if i == 0:
                decoder_out, decoder_state = self.decoder(initial_input, state)

            else:
                current_input = y[:, i - 1 : i, :, :] if y is not None else decoder_out

                decoder_out, decoder_state = self.decoder(current_input, decoder_state)
            decoder_output_list.append(decoder_out)

        return decoder_output_list

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch

        prev_sequence_raw = x.view(
            -1,
            self.satellite_channels,
            self.input_image_size[0],
            self.input_image_size[1],
        )
        prev_sequence_encoded = self.cnn_encoder(prev_sequence_raw).view(
            1, 5, 64, self.model_img_dim, self.model_img_dim
        )
        state = self.encode_temporal(prev_sequence_encoded)

        decoder_output_list = self.decode_temporal(state)

        final_decoder_out = torch.cat(decoder_output_list, 1)

        decoder_out_rs = final_decoder_out.view(
            -1, 64, self.model_img_dim, self.model_img_dim
        )
        result = self.cnn_decoder(decoder_out_rs)

        y_hat = result.view(1, 12, 166, 134)

        return y_hat


@step(
    enable_cache=False,
    experiment_tracker="Infoplaza MLFlow",
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader, val_dataloader: DataLoader) -> nn.Module:
    model = Sat2Rad()
    # trainer = pl.Trainer(
    #     max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
    # )
    trainer = pl.Trainer(max_epochs=100)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # fetch the auto logged parameters and metrics
    # log_utils.log_mlflow(run=mlflow.get_run(run_id=run.info.run_id))

    return model


if __name__ == "__main__":
    model = Sat2Rad()

    x = torch.randn((1, 5, 4, 166, 134))
    y = torch.randn((1, 12, 1, 166, 134))

    dummyBatch = (x, y)

    loss = model.training_step(dummyBatch, 1)

    print(loss)
