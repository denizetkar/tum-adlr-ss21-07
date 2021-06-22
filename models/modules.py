from typing import List, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces


class TuplePick(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, *t_args: th.Tensor) -> th.Tensor:
        return t_args[self.index]

    def extra_repr(self) -> str:
        return "(index): {}".format(self.index)


class TupleApply(nn.Module):
    def __init__(self, *mods: nn.Module):
        super().__init__()
        self.mods = nn.Sequential(*mods)

    def forward(self, *t_args: th.Tensor) -> Tuple[th.Tensor, ...]:
        return tuple(mod(x) for mod, x in zip(self.mods, t_args))


class Reshape(nn.Module):
    def __init__(self, *args: int):
        super().__init__()
        self.shape = args

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x.reshape(*self.shape)

    def extra_repr(self) -> str:
        return "(shape): {}".format(self.shape)


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

    def extra_repr(self) -> str:
        return "(nvec): {}".format(self.nvec)
