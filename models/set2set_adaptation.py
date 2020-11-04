import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FEATTransformer(nn.Module):
    def __init__(self, n_features, dropout=0.5):
        super(FEATTransformer, self).__init__()

        self.n_features = n_features
        self.dropout = dropout

        self.q_linear = nn.Linear(self.n_features, self.n_features, bias=False)
        self.k_linear = nn.Linear(self.n_features, self.n_features, bias=False)
        self.v_linear = nn.Linear(self.n_features, self.n_features, bias=False)

        nn.init.normal_(self.q_linear.weight, mean=0, std=np.sqrt(1.0 / n_features))
        nn.init.normal_(self.k_linear.weight, mean=0, std=np.sqrt(1.0 / n_features))
        nn.init.normal_(self.v_linear.weight, mean=0, std=np.sqrt(1.0 / n_features))

        self.layer_norm = nn.LayerNorm(n_features)

        self.fc = nn.Linear(self.n_features, self.n_features)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q_expanded = q.repeat(batch_size, 1)
        k_expanded = k.repeat_interleave(batch_size, dim=0)
        v_expanded = v.repeat(batch_size, 1).view(batch_size, batch_size, -1)

        attention = F.softmax(
            ((q_expanded * k_expanded).sum(dim=1) / math.sqrt(self.n_features)).view(batch_size, batch_size), dim=1)

        x_adapted = (attention.unsqueeze(2) * v_expanded).sum(dim=1)

        output = self.fc(x_adapted)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output
