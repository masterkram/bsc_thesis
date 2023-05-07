import torch
import torch.nn as nn
from .layers.ConvLSTM import ConvLSTM

b_size = 32

hidden_dim = 64
hidden_spt = 16

lstm_dims = [64, 64, 64]


class Sat2Rad(nn.Module):
    def __init__(self):
        super(Sat2Rad, self).__init__()
        self.conv_lstm = ConvLSTM(
            input_size=(hidden_spt, hidden_spt), input_dim=hidden_dim, hidden_dim=lstm_dims, kernel_size=(3, 3), num_layers=3)

    def forward(self, x: torch.Tensor):
        hidden = self.conv_lstm.get_init_states(b_size, cuda=False)
        return self.conv_lstm(x, hidden)


if __name__ == "main":
    print("hello mark")
