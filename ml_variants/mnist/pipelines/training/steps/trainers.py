from zenml.steps import step
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import mlflow
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)
import lightning.pytorch as pl
from torchmetrics.functional import accuracy

"""
    Local imports
"""
from utils import log_utils

mlflow_settings = MLFlowExperimentTrackerSettings(experiment_name="mnist_pytorch_test")


class ImageClassifier(pl.LightningModule):
    def __init__(
        self, conv_1: dict[str, int], conv_2: dict[str, int], linear: dict[str, int]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(
                conv_1["in_channels"],
                conv_1["filters"],
                (conv_1["kernel_size"], conv_1["kernel_size"]),
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_2["in_channels"],
                conv_2["filters"],
                (conv_2["kernel_size"], conv_2["kernel_size"]),
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear["in_features"], linear["out_features"]),
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
def trainer(train_data: DataLoader) -> pl.LightningModule:
    """
    Train model.

    Arguments:
    + train_dataloader (`DataLoader`): training partition of the dataset.

    Returns:
    + model (`LightningModule`): Trained Model.
    """
    mlflow.end_run()

    conv_layer_1 = {"in_channels": 1, "filters": 32, "kernel_size": 3}
    conv_layer_2 = {"in_channels": 32, "filters": 64, "kernel_size": 3}
    linear_layer = {"in_features": 36864, "out_features": 10}

    model = ImageClassifier(
        conv_1=conv_layer_1, conv_2=conv_layer_2, linear=linear_layer
    )

    trainer = pl.Trainer(max_epochs=15)

    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        mlflow.log_params(model.hparams)
        trainer.fit(model, train_data)

    # fetch the auto logged parameters and metrics
    log_utils.log_mlflow(run=mlflow.get_run(run_id=run.info.run_id))

    return model
