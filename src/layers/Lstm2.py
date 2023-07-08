import torch
import torch.nn as nn


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(
            in_channels + num_features, num_features * 4, kernel_size, padding, stride
        )

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs):
        """

        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        """
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], hx], dim=1)  # (B, C, H, W)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1
            )
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return (
            torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
        )  # (S, B, C, H, W) -> (B, S, C, H, W)
