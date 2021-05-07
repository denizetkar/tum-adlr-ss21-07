import math
from typing import List, Tuple, Type

import torch as th
import torch.nn as nn

from .lstm import LSTM


class MultiLayerLSTM(nn.Module):
    """
    A helper class to contruct multi-layered LSTMs.
    It can build multiLayer LSTM of any type.

    Note: Dropout is deactivated on the last layer.
    """

    def __init__(self, input_size: int, layer_type: Type[LSTM], layer_sizes: List[int], *args, **kwargs):
        super().__init__()
        self.layers: List[LSTM] = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            layer = layer_type(input_size=prev_size, hidden_size=size, *args, **kwargs)
            self.layers.append(layer)
            prev_size = size
        if "dropout" in kwargs:
            del kwargs["dropout"]
        if len(layer_sizes) > 0:
            layer = layer_type(input_size=prev_size, hidden_size=layer_sizes[-1], dropout=0.0, *args, **kwargs)
            self.layers.append(layer)
        self.params = nn.ModuleList(self.layers)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    @staticmethod
    def _hidden_state_name(layer: int, postfix: str = "h") -> str:
        return f"layer_{layer}_{postfix}"

    def get_hidden_state(self, layer: int) -> Tuple[th.Tensor, th.Tensor]:
        h = getattr(self, MultiLayerLSTM._hidden_state_name(layer, "h"))
        c = getattr(self, MultiLayerLSTM._hidden_state_name(layer, "c"))
        return h, c

    def set_hidden_state(self, layer: int, hidden_state: Tuple[th.Tensor, th.Tensor]):
        new_h, new_c = hidden_state
        self.register_buffer(MultiLayerLSTM._hidden_state_name(layer, "h"), new_h)
        self.register_buffer(MultiLayerLSTM._hidden_state_name(layer, "c"), new_c)

    def reset_hiddens(self, batch_size: int = 1):
        device = next(self.layers[0].parameters()).device
        # Uses Xavier init here.
        for idx, layer in enumerate(self.layers):
            std = math.sqrt(2.0 / (layer.input_size + layer.hidden_size))
            self.set_hidden_state(
                idx,
                (
                    th.empty((batch_size, layer.hidden_size), device=device, requires_grad=False).normal_(0, std),
                    th.empty((batch_size, layer.hidden_size), device=device, requires_grad=False).normal_(0, std),
                ),
            )

    def reset_batch_hiddens(self, batch_idx: int):
        # Uses Xavier init here.
        for idx, layer in enumerate(self.layers):
            std = math.sqrt(2.0 / (layer.input_size + layer.hidden_size))
            h, c = self.get_hidden_state(idx)
            index_tensor = th.tensor([batch_idx]).long().to(h.device)
            new_h = h.index_copy(0, index_tensor, th.empty(layer.hidden_size).normal_(0, std).unsqueeze_(0).to(h.device))
            new_c = c.index_copy(0, index_tensor, th.empty(layer.hidden_size).normal_(0, std).unsqueeze_(0).to(c.device))
            self.set_hidden_state(idx, (new_h, new_c))

    def sample_mask(self):
        for layer in self.layers:
            layer.sample_mask()

    def forward(self, x: th.Tensor) -> th.Tensor:
        for idx, layer in enumerate(self.layers):
            x, (new_h, new_c) = layer(x, self.get_hidden_state(idx))
            self.set_hidden_state(idx, (new_h, new_c))
        return x
