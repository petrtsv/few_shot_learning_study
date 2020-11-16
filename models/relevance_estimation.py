import torch
from torch import nn


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
    )


class RelevanceEstimator(nn.Module):
    def __init__(self, in_dim, in_map_size, hidden_dim=128):
        super(RelevanceEstimator, self).__init__()

        self.conv1 = conv_block(in_dim, hidden_dim)
        self.conv2 = conv_block(hidden_dim, 2 * hidden_dim)

        self.fc = nn.Linear(in_features=2 * hidden_dim * in_map_size * in_map_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))
