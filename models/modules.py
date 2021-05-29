from typing import List, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces


class MLP(nn.Module):
    def __init__(self, input_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.PReLU):
        super().__init__()
        layers: List[nn.Module] = []
        for hidden_dim in net_arch[:-1]:
            layers.extend([nn.Linear(input_dim, hidden_dim), activation_fn()])
            input_dim = hidden_dim
        if len(net_arch) > 0:
            layers.append(nn.Linear(input_dim, net_arch[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.mlp(x)

    def __repr__(self):
        return "{}([{}])".format(self.__class__.__name__, ", ".join([repr(self.mlp[i]) for i in range(len(self.mlp))]))


class TuplePick(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, *t_args: th.Tensor) -> th.Tensor:
        return t_args[self.index]

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.index)


class TupleApply(nn.Module):
    def __init__(self, *mods: nn.Module):
        super().__init__()
        self.mods = nn.Sequential(*mods)

    def forward(self, *t_args: th.Tensor) -> Tuple[th.Tensor, ...]:
        return tuple(mod(x) for mod, x in zip(self.mods, t_args))

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join([repr(mod) for mod in self.mods]))


class Reshape(nn.Module):
    def __init__(self, *args: int):
        super().__init__()
        self.shape = args

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x.reshape(*self.shape)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join([repr(dim) for dim in self.shape]))


class SequentialExpand(nn.Module):
    def __init__(self, *mods: nn.Module):
        super().__init__()
        self.mods = nn.Sequential(*mods)

    def forward(self, *t_args: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, ...]]:
        for mod in self.mods:
            if isinstance(t_args, th.Tensor):
                t_args = mod(t_args)
            else:
                t_args = mod(*t_args)
        return t_args


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, action_space: spaces.MultiDiscrete):
        super().__init__()
        if isinstance(action_space.nvec, np.ndarray):
            self.nvec = action_space.nvec.tolist()
        else:
            self.nvec: List[int] = action_space.nvec
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs: th.Tensor, targets: th.Tensor) -> th.Tensor:
        # 'inputs' is of shape (n, sum(self.nvec)) and 'targets' is of shape (n, len(self.nvec))
        losses = th.cat(
            [
                self.ce_loss(inp, target)
                for inp, target in zip(th.split(inputs, self.nvec, dim=-1), th.split(targets, 1, dim=-1))
            ]
        )
        return losses.mean()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join([repr(dim) for dim in self.nvec]))
