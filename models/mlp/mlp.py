from typing import List

import torch as th
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, feature_dim: int, net_arch: List[int], activation_fn: nn.Module = nn.PReLU, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for hidden_dim in net_arch[:-1]:
            layers.append(nn.Linear(feature_dim, hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            feature_dim = hidden_dim
        if len(net_arch) > 0:
            layers.append(nn.Linear(feature_dim, net_arch[-1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)
