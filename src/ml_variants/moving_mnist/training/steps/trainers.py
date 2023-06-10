import sys
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

sys.path.append("../../../../")
sys.path.append("../../../")

from zenml import step
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch import optim
import mlflow
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
    def __init__(self, in_channels=settings.input_size.channels, out_dim=64):
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
            nn.ReLU(),
        )

    def forward(self, input):
        output = self.cnn_encoder(input)
        return output


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
            input_dim=settings.encoder.filters,
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

        self.loss_fn = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        seqs = batch
        nextf_raw = seqs[:, 10:, :, :, :]

        prevf_raw = seqs[:, :10, :, :, :].contiguous().view(-1, 1, 64, 64)

        prev_sequence_encoded = self.cnn_encoder(prevf_raw).view(
            1, 10, 64, self.model_img_dim[0], self.model_img_dim[1]
        )

        cnn_encoder_out = self.cnn_encoder(
            seqs[:, 10:, :, :, :].contiguous().view(-1, 1, 64, 64).cuda()
        )
        nextf_enc = cnn_encoder_out.view(32, 10, 64, 16, 16)

        state = self.encode_temporal(prev_sequence_encoded)

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

        prev_sequence_raw = x.view(
            -1,
            self.satellite_channels,
            self.input_image_size[0],
            self.input_image_size[1],
        )
        prev_sequence_encoded = self.cnn_encoder(prev_sequence_raw).view(
            1, 10, 64, self.model_img_dim[0], self.model_img_dim[1]
        )
        state = self.encode_temporal(prev_sequence_encoded)

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
    # trainer = pl.Trainer(
    #     max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
    # )
    trainer = pl.Trainer(max_epochs=9)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # fetch the auto logged parameters and metrics
    # log_utils.log_mlflow(run=mlflow.get_run(run_id=run.info.run_id))

    return model
