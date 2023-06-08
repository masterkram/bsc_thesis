import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from zenml import step


@step
def evaluator(test_dataloader: DataLoader, model: nn.Module) -> float:
    """Evaluates on the model."""
    trainer = pl.Trainer()
    result = trainer.test(model, test_dataloader)
    return result[0]["test_loss"]
