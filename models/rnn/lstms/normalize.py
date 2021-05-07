"""
Implementation of various normalization techniques. Also only works on instances
where batch size = 1.
"""
from typing import Iterable

import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size: int, learnable: bool = True, epsilon: float = 1e-6):
        super().__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.weights = th.empty(1, input_size)
        self.biases = th.empty(1, input_size)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.weights = W(self.weights)
        self.biases = W(self.biases)
        self.reset_parameters()

    def reset_parameters(self):
        for name, w in self.named_parameters():
            if "weight" in name:
                nn.init.constant_(w, 1.0)
            elif "bias" in name:
                nn.init.constant_(w, 0.0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).unsqueeze(1)) / th.sqrt(th.var(x, 1).unsqueeze(1) + self.epsilon)
        if self.learnable:
            x = self.weights.expand_as(x) * x + self.biases.expand_as(x)
        return x.view(size)


class BradburyLayerNorm(nn.Module):

    """
    Layer Norm, according to:
    https://github.com/pytorch/pytorch/issues/1959#issuecomment-312364139
    """

    def __init__(self, features: Iterable[int], eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(th.ones(features))
        self.beta = nn.Parameter(th.zeros(features))
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class BaLayerNorm(nn.Module):

    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    This implementation mimicks the original torch implementation at:
    https://github.com/ryankiros/layer-norm/blob/master/torch_modules/LayerNormalization.lua
    """

    def __init__(self, input_size: int, learnable: bool = True, epsilon: float = 1e-5):
        super().__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.epsilon = epsilon
        self.alpha = th.empty(1, input_size).fill_(0)
        self.beta = th.empty(1, input_size).fill_(0)
        # Wrap as parameters if necessary
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)

    def forward(self, x: th.Tensor) -> th.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        mean = th.mean(x, 1).expand_as(x)
        center = x - mean
        std = th.sqrt(th.mean(th.square(center), 1)).expand_as(x)
        output = center / (std + self.epsilon)
        if self.learnable:
            output = self.alpha * output + self.beta
        return output.view(size)
