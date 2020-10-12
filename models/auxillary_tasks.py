from typing import Tuple

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


class RotationClassifier(nn.Module):
    def __init__(self, in_dim, in_map_size, hidden_dim=128):
        super(RotationClassifier, self).__init__()

        self.conv1 = conv_block(in_dim, hidden_dim)
        self.conv2 = conv_block(hidden_dim, 2 * hidden_dim)

        self.fc = nn.Linear(in_features=2 * hidden_dim * in_map_size * in_map_size, out_features=4)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RotationTask(nn.Module):
    def __init__(self, in_features: int, in_features_size: int):
        super(RotationTask, self).__init__()

        self.solver = RotationClassifier(in_dim=in_features,
                                         in_map_size=in_features_size)

        self.loss_fn = nn.CrossEntropyLoss()

    def get_task(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch.size(0)
        target = torch.randint(4, (batch_size,)).to(batch.device)
        for i in range(batch_size):
            batch[i] = torch.rot90(batch[i], target[i].item(), dims=(1, 2))

        return batch, target

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.solver(features)

        return self.loss_fn(output, target)
