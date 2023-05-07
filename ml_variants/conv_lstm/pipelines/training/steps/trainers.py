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


@step(
    enable_cache=False,
    experiment_tracker="Infoplaza MLFlow",
    settings={"experiment_tracker.mlflow": mlflow_settings},
)
def trainer(train_dataloader: DataLoader) -> nn.Module:
    clf = ImageClassifier().to(DEVICE)
    opt = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(10):
        for batch in train_dataloader:
            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"epoch {epoch} loss: {loss.item()}")
    return clf
