from zenml.steps import step
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import mlflow
from mlflow import MlflowClient
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name="mnist_pytorch_test")


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*24*24, 10)
        )

    def forward(self, x):
        return self.model(x)


@step(enable_cache=False, experiment_tracker="Infoplaza MLFlow", settings={'experiment_tracker.mlflow': mlflow_settings})
def trainer(train_dataloader: DataLoader) -> nn.Module:
    clf = ImageClassifier().to('cpu')
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
            print(f'epoch {epoch} loss: {loss.item()}')
    return clf
