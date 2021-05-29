from typing import NamedTuple

import torch as th


class EpisodicRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    returns: th.Tensor
    dones: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    batch_inds: th.Tensor
