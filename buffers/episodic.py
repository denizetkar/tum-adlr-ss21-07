from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

from .samples import EpisodicRolloutBufferSamples

# + Minibatches over episodes instead of over individual experiences (for RNN),
#   * Batch size will still be in number of timesteps instead of number of episodes so that
#     each minibatch contains more or less equal experience.
#   * Modify RolloutBuffer.get() method for implementation.
#   * Pass "_init_setup_model=False" as parameter to PPO.__init__() method and do the initialization
#     yourself (self._setup_model()) so that you can use your own rollout buffer.
#   * Make sure to also return "dones" in minibatches to distinguish episode borders during
#     RNN training -> reimplement RolloutBufferSamples class.


NON_SCALAR_DATA_FIELDS = ["observations", "actions"]
SCALAR_DATA_FIELDS = ["rewards", "returns", "dones", "values", "log_probs", "advantages"]


class EpisodeBoundary(NamedTuple):
    from_idx: int
    to_idx: int


class EpisodicRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self._ep_boundaries: List[EpisodeBoundary] = []
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self._env_infos: List[Dict[str, Any]] = [{"r": 0.0} for _ in range(self.dones.shape[1])]

    def unflatten_and_swap(self, arr: np.ndarray) -> np.ndarray:
        """
        Does the opposite of 'swap_and_flatten' method. Unflattens the first axis
        [n_envs*n_steps, ...] into [n_envs, n_steps, ...]. Then swaps the first 2
        axes into [n_steps, n_envs, ...]

        :param arr:
        :return:
        """
        shape = arr.shape
        return arr.reshape(self.n_envs, -1, *shape[1:]).swapaxes(0, 1)

    def soft_reset(self):
        # Unflattens and swaps the data fields without resetting them.
        for tensor in NON_SCALAR_DATA_FIELDS + SCALAR_DATA_FIELDS:
            self.__dict__[tensor] = self.unflatten_and_swap(self.__dict__[tensor])
        for tensor in SCALAR_DATA_FIELDS:
            self.__dict__[tensor] = np.squeeze(self.__dict__[tensor], axis=-1)
        self.generator_ready = False

    def reset(self) -> None:
        self._ep_boundaries.clear()
        super().reset()

    def get(self, min_batch_size: Optional[int] = None) -> Generator[EpisodicRolloutBufferSamples, None, None]:
        assert self.full, ""
        # Prepare the data
        if not self.generator_ready:
            dones = self.dones.copy()
            dones[0, :] = 1.0
            for tensor in NON_SCALAR_DATA_FIELDS + SCALAR_DATA_FIELDS:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self._find_ep_boundaries(self.swap_and_flatten(dones).astype(dtype=bool).reshape(-1))
            self.generator_ready = True
        # Randomize over episodes instead of over individual experiences
        ep_indices: List[int] = np.random.permutation(len(self._ep_boundaries)).tolist()

        # Return everything, don't create minibatches
        if min_batch_size is None:
            min_batch_size = self.buffer_size * self.n_envs

        ep_idx = 0
        while ep_idx < len(self._ep_boundaries):
            indices, ep_idx = self._get_batch_indices(ep_indices, ep_idx, min_batch_size)
            yield self._get_samples(indices)

    def _find_ep_boundaries(self, dones: np.ndarray):
        self._ep_boundaries.clear()
        from_idx, to_idx = 0, 0
        for done in dones:
            if done:
                if from_idx < to_idx:
                    self._ep_boundaries.append(EpisodeBoundary(from_idx, to_idx))
                from_idx = to_idx
            to_idx += 1
        if from_idx < to_idx:
            self._ep_boundaries.append(EpisodeBoundary(from_idx, to_idx))

    def _get_batch_indices(self, ep_indices: List[int], ep_idx: int, min_batch_size: int) -> Tuple[np.ndarray, int]:
        last_ep_idx = ep_idx
        indices: List[int] = []
        while last_ep_idx < len(self._ep_boundaries):
            indices.extend(
                range(
                    self._ep_boundaries[ep_indices[last_ep_idx]].from_idx,
                    self._ep_boundaries[ep_indices[last_ep_idx]].to_idx,
                )
            )
            last_ep_idx += 1
            if len(indices) >= min_batch_size:
                break
        return np.array(indices), last_ep_idx

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> EpisodicRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.dones[batch_inds].flatten(),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            batch_inds,
        )
        return EpisodicRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def get_episode_infos(self) -> List[Dict[str, Any]]:
        # This method assumes that buffers are not swapped and flattened!
        ep_infos: List[Dict[str, Any]] = []
        for step in range(self.dones.shape[0]):
            for env_index, env_info in enumerate(self._env_infos):
                if self.dones[step, env_index]:
                    ep_infos.append(env_info.copy())
                    env_info["r"] = float(self.rewards[step, env_index])
                    continue
                env_info["r"] += self.rewards[step, env_index]
        return ep_infos
