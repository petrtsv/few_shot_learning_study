from typing import Dict, List

import torch
from torch import nn

from utils import inverse_mapping


class ScaleModule(nn.Module):
    def __init__(self, in_features, map_size):
        super(ScaleModule, self).__init__()
        self.in_features = in_features
        self.conv = nn.Conv2d(in_channels=self.in_features, out_channels=1, kernel_size=3)
        self.bn = nn.BatchNorm2d(1, eps=2e-5)
        self.relu = nn.ReLU()

        self.fc = nn.Linear((map_size - 2) ** 2, 1)
        self.sp = nn.Softplus()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.sp(x)

        return x


class DFMNLoss(nn.Module):
    def __init__(self, in_features: int, in_featmap_size: int, n_classes: int):
        super(DFMNLoss, self).__init__()

        self.in_features = in_features
        self.in_featmap_size = in_featmap_size
        self.n_classes = n_classes
        self.global_prototypes = nn.Linear(in_features=self.in_features, out_features=self.n_classes)
        nn.init.xavier_uniform_(self.global_prototypes.weight)

        self.loss_fn = nn.CrossEntropyLoss()

    def get_l2_distances(self, query_set: torch.Tensor, prototypes: torch.Tensor):
        cur_n_classes = prototypes.size(0)
        cur_query_set_size = query_set.size(0)

        query_set_expanded = query_set.repeat_interleave(cur_n_classes, dim=0)
        prototypes_expanded = prototypes.repeat(cur_query_set_size, 1)
        distances = (query_set_expanded - prototypes_expanded).pow(2).sum(dim=1)
        distances = torch.stack(distances.split(cur_n_classes))
        return -distances

    def forward(self, query_features: torch.Tensor, labels: torch.Tensor, global_classes_mapping: Dict[int, int],
                n_way: int) -> torch.Tensor:
        featmap_size_sq = int(self.in_featmap_size * self.in_featmap_size)

        expanded_labels = labels.clone().repeat_interleave(featmap_size_sq, dim=0)
        prototypes = self.global_prototypes.weight

        inv_mapping = inverse_mapping(global_classes_mapping)

        indices_list: List[int] = []
        for i in range(n_way):
            indices_list.append(inv_mapping[i])
        indices = torch.tensor(indices_list, device=prototypes.device)
        # print(indices)

        prototypes = torch.index_select(prototypes, 0, indices)
        # print(prototypes.size())

        expanded_query_features = query_features.view(-1, self.in_features, self.in_featmap_size, self.in_featmap_size)
        expanded_query_set = expanded_query_features.permute(0, 2, 3, 1).reshape((-1, self.in_features))
        # print(expanded_query_set.size())

        distances = self.get_l2_distances(expanded_query_set, prototypes)
        # print(distances.size())
        return self.loss_fn(distances, expanded_labels)
