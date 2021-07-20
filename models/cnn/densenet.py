from typing import Type

import torch as th
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, is_last_layer: bool = False):
        super().__init__()
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        ]
        if not is_last_layer:
            layers.extend([nn.PReLU(), nn.Dropout(dropout)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Adjust batch normalization modules in case the batch size is 1.
        if x.shape[1] == 1:
            self.layers[0].eval()
        else:
            self.layers[0].train()
        out = self.layers(x)
        return th.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, layer_cnt: int, in_channels: int, growth_rate: int, block_type: Type[nn.Module], dropout: float = 0.0):
        super().__init__()
        self.layers = self._make_layers(block_type, in_channels, growth_rate, layer_cnt, dropout)

    def _make_layers(self, block_type: Type[nn.Module], in_channels: int, growth_rate: int, layer_cnt: int, dropout: float):
        layers = []
        for i in range(layer_cnt - 1):
            layers.append(block_type(in_channels + i * growth_rate, growth_rate, dropout))
        if layer_cnt > 0:
            layers.append(block_type(in_channels + (layer_cnt - 1) * growth_rate, growth_rate, dropout, True))
        return nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)
