import sys
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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

# from utils import log_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import typed_settings as ts
from Settings import ModelSettings

settings = ts.load(ModelSettings, "model", ["config.toml"])


mlflow_settings = MLFlowExperimentTrackerSettings(experiment_name="sat2rad_conv_lstm")


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=(3, 3),
                stride=1,
                padding="same",
            ),
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
                in_channels=self.in_channels, out_channels=16, kernel_size=(3, 3)
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.out_dim, kernel_size=(3, 3)),
        )

    def forward(self, input):
        output = self.cnn_encoder(input)
        return output


class ConvEncoderRadar(nn.Module):
    def __init__(self, out_dim=64):
        super(ConvEncoderRadar, self).__init__()
        self.out_dim = out_dim
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=self.out_dim,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1
            ),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=(6, 107), stride=1
            ),
        )

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


class Weighted_mse_mae(nn.Module):
    def __init__(
        self,
        mse_weight=1.0,
        mae_weight=1.0,
        NORMAL_LOSS_GLOBAL_SCALE=0.00005,
        LAMBDA=None,
    ):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target):
        balancing_weights = (1, 1, 2, 5, 10, 30)
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [
            ele
            for ele in np.array(
                [0.104712042, 0.209424084, 0.314136126, 0.418848168, 0.523560209]
            )
        ]
        for i, threshold in enumerate(thresholds):
            weights = (
                weights
                + (balancing_weights[i + 1] - balancing_weights[i])
                * (target >= threshold).float()
            )
        # weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input - target) ** 2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input - target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (
            self.mse_weight * torch.mean(mse) + self.mae_weight * torch.mean(mae)
        )


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.radar_sequence_length = settings.output_size.sequence_length
        self.satellite_channels = settings.input_size.sequence_length
        self.input_image_size = (settings.input_size.height, settings.input_size.width)
        self.output_image_size = (
            settings.output_size.height,
            settings.output_size.width,
        )

        self.model_img_dim = (settings.output_size.height, settings.output_size.width)
        hidden_img = (settings.output_size.height, settings.output_size.width)
        hidden_dim = [settings.encoder.filters for _ in range(settings.encoder.layers)]
        kernel_size = (3, 3)
        self.channels = settings.encoder.filters

        self.encoder = ConvLSTM(
            input_size=hidden_img,
            input_dim=11,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=settings.encoder.layers,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )

        self.decoder = ConvLSTM(
            input_size=hidden_img,
            input_dim=settings.decoder.filters,
            hidden_dim=hidden_dim,
            kernel_size=tuple(settings.decoder.kernel_size),
            num_layers=settings.decoder.layers,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )

        self.cnn_encoder = ConvEncoder(
            in_channels=self.satellite_channels, out_dim=self.channels
        )
        self.cnn_decoder = ConvDecoder()
        self.encoder_radar = ConvEncoderRadar()

        self.loss_fn = Weighted_mse_mae()
        # self.loss_fn = balanced_mae

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(1, 12, 1, 166, 134)
        # y passed => teacher forcing
        # seq_out = y[:, :, :, :].detach().clone()
        # cnn_encoder_out = self.encoder_radar(
        #     seq_out.view(-1, 1, self.output_image_size[0], self.output_image_size[1])
        # )
        # print(cnn_encoder_out.size())
        # nextf_enc = cnn_encoder_out.view(
        #     1, 12, self.channels, self.model_img_dim[0], self.model_img_dim[1]
        # )

        # prev_sequence_raw = x.view(
        #     -1,
        #     self.satellite_channels,
        #     self.input_image_size[0],
        #     self.input_image_size[1],
        # )
        # prev_sequence_encoded = self.cnn_encoder(prev_sequence_raw).view(
        #     1, 5, 64, self.model_img_dim[0], self.model_img_dim[1]
        # )

        state = self.encode_temporal(x)

        decoder_output_list = self.decode_temporal(state=state)

        final_decoder_out = torch.cat(decoder_output_list, 1)

        decoder_out_rs = final_decoder_out.view(
            -1, self.channels, self.model_img_dim[0], self.model_img_dim[1]
        )
        result = self.cnn_decoder(decoder_out_rs)

        y_hat = result.view(
            1,
            self.radar_sequence_length,
            self.output_image_size[0],
            self.output_image_size[1],
        )

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
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def encode_temporal(self, x: torch.Tensor):
        hidden = self.encoder.get_init_states(batch_size=1, device=self.device)
        _, encoder_state = self.encoder(x, hidden)
        return encoder_state

    def decode_temporal(self, state: torch.Tensor, y=None):
        decoder_output_list = []

        initial_input = torch.zeros(
            (1, 1, self.channels, self.model_img_dim[0], self.model_img_dim[1])
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

        # prev_sequence_raw = x.view(
        #     -1,
        #     self.satellite_channels,
        #     self.input_image_size[0],
        #     self.input_image_size[1],
        # )
        # prev_sequence_encoded = self.cnn_encoder(prev_sequence_raw).view(
        #     1, 5, 64, self.model_img_dim[0], self.model_img_dim[1]
        # )
        state = self.encode_temporal(x)

        decoder_output_list = self.decode_temporal(state)

        final_decoder_out = torch.cat(decoder_output_list, 1)

        decoder_out_rs = final_decoder_out.view(
            -1, 64, self.model_img_dim[0], self.model_img_dim[1]
        )
        result = self.cnn_decoder(decoder_out_rs)

        y_hat = result.view(
            1,
            self.radar_sequence_length,
            self.output_image_size[0],
            self.output_image_size[1],
        )

        return y_hat


@step(
    enable_cache=False,
    experiment_tracker="Infoplaza MLFlow",
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader, val_dataloader: DataLoader) -> nn.Module:
    model = Sat2Rad()
    trainer = pl.Trainer(max_epochs=100)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # fetch the auto logged parameters and metrics
    # log_utils.log_mlflow(run=mlflow.get_run(run_id=run.info.run_id))

    return model
