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
from utils import log_utils
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mlflow_settings = MLFlowExperimentTrackerSettings(experiment_name="sat2rad_conv_lstm")


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            ConvLSTM(
                input_size=(64, 64),
                input_dim=1,
                hidden_dim=1,
                kernel_size=(2, 2),
                num_layers=4,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 10),
        )

    def forward(self, x):
        return self.model(x)


b_size = 32

hidden_dim = 64
hidden_spt = 16

lstm_dims = [64, 64, 64]

teacher_forcing = True


class Sat2Rad(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            ConvLSTM(
                input_size=(hidden_spt, hidden_spt),
                input_dim=hidden_dim,
                hidden_dim=lstm_dims,
                kernel_size=(3, 3),
                num_layers=3,
                peephole=True,
                batchnorm=False,
                batch_first=True,
                activation=F.tanh,
            ),
            nn.ReLU(),
            nn.Conv2d(
                64,
                1,
                (3, 3),
            ),
            nn.ReLU(),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)
        acc = self.calculate_accuracy(yhat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)
        acc = self.calculate_accuracy(yhat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def calculate_accuracy(yhat, y):
        return accuracy(yhat, y, task="multiclass", num_classes=10)


@step(
    enable_cache=False,
    experiment_tracker="Infoplaza MLFlow",
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader) -> nn.Module:
    model = Sat2Rad()
    trainer = pl.Trainer(max_epochs=10)

    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        mlflow.log_params(model.hparams)
        trainer.fit(model, train_dataloader)

    # fetch the auto logged parameters and metrics
    log_utils.log_mlflow(run=mlflow.get_run(run_id=run.info.run_id))

    return model
