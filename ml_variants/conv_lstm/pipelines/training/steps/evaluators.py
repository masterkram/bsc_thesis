import torch
from torch import nn
from torch.utils.data import DataLoader

from zenml.steps import step


@step
def evaluator(test_dataloader: DataLoader, model: nn.Module) -> float:
    """Evaluates on the model."""
    # pass
    return 0.5
