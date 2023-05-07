import layers.ConvLSTM
from zenml.steps import step
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
        


@step(enable_cache=False)
def train_conv_lstm(X_train: np.ndarray, y_train: np.ndarray) -> nn.Module:
    