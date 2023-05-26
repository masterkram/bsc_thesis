import sys

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


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ConvLSTM(
            input_size=(166, 134),
            input_dim=4,
            hidden_dim=[64],
            kernel_size=(3, 3),
            num_layers=1,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )

        self.decoder = ConvLSTM(
            input_size=(166, 134),
            input_dim=64,
            hidden_dim=[1],
            kernel_size=(3, 3),
            num_layers=1,
            peephole=True,
            batchnorm=False,
            batch_first=True,
            activation=F.tanh,
        )

        self.conv_1 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
        )

        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=1
        )

        self.conv_3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=(3, 3), stride=1, padding=1
        )

        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        _, y = batch
        y_hat = self(batch)

        # y = y.view(1, 12, 1, 166, 134)
        loss = self.loss_fn(y_hat, y)
        return loss
        # acc = self.calculate_accuracy(yhat, y)

        # self.log("train_loss", loss, on_epoch=True)
        # self.log("acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        _, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        return loss

        # self.log("train_loss", loss, on_epoch=True)
        # self.log("acc", acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    @staticmethod
    def calculate_accuracy(yhat, y):
        return accuracy(yhat, y, task="multiclass", num_classes=10)

    # def encode_temporal(self, batch: tuple[torch.Tensor, torch.Tensor]):
    #     hidden = self.encoder.get_init_states(batch_size=1, device=self.device)
    #     _, encoder_state = self

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch

        # ------ lstm encoder -----
        hidden = self.encoder.get_init_states(batch_size=1, device=self.device)
        encoder_out, encoder_state = self.encoder(x, hidden)

        # ------ lstm decoder -----
        decoder_output_list = []

        y = y.view(1, 12, 1, 166, 134)

        for i in range(12):
            if i == 0:
                state = encoder_state
                decoder_out, decoder_state = self.decoder(
                    y[:, i : i + 1, :, :, :], state
                )

            else:
                state = decoder_state
                decoder_out, decoder_state = self.decoder(
                    y[:, i : i + 1, :, :, :], state
                )

            decoder_output_list.append(decoder_out)

        final_decoder_out = torch.cat(decoder_output_list, 1)

        # ------ decoding convolutions -----
        decoder_out_rs = final_decoder_out.view(-1, 64, 166, 134)
        result = self.conv_1(decoder_out_rs)
        result = self.conv_2(result)
        result = self.conv_3(result)

        result = result.view(1, 12, 166, 134)

        return result


@step(
    enable_cache=False,
    experiment_tracker="Infoplaza MLFlow",
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader) -> nn.Module:
    mlflow.end_run()
    model = Sat2Rad()
    trainer = pl.Trainer(max_epochs=10)

    mlflow.pytorch.autolog()

    mlflow.log_params(model.hparams)
    trainer.fit(model, train_dataloader)

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
