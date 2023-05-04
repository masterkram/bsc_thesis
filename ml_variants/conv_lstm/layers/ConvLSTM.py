import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTMCell(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int,
            bias=True,
            activation=torch.tanh,
            batchnorm=False,
    ):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.activation = activation
        self.batchnorm = batchnorm

    def forward(self, x: torch.Tensor, prev_state: list) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = prev_state

        combined = torch.cat((x, h_prev), dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)


class ConvLSTM(nn.Module):
    def __init__():
        super(ConvLSTM, self).__init__()
