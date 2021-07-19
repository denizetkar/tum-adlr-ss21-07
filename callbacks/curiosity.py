import os
from typing import List, Optional, Union

import gym
import torch as th
import torch.nn as nn
from buffers.episodic import EpisodicRolloutBuffer
from gym import spaces
from models import CuriosityModel, MultiCrossEntropyLoss
from stable_baselines3.common import logger
from stable_baselines3.common.utils import get_device

from .base import EnhancedBaseCallback


class CuriosityCallback(EnhancedBaseCallback):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float = 3e-4,
        n_epochs: int = 3,
        curiosity_coefficient: float = 0.5,
        latent_dim: int = 64,
        partially_observable: bool = True,
        pure_curiosity_reward: bool = False,
        idm_net_arch: Optional[List[int]] = None,
        forward_net_arch: Optional[List[int]] = None,
        model_path: Optional[str] = None,
        device: Union[th.device, str] = "auto",
        alternate_train: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.action_space = action_space
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.curiosity_coefficient = curiosity_coefficient
        self.pure_curiosity_reward = pure_curiosity_reward
        self.model_path = model_path
        self.device = get_device(device)
        self.alternate_train = alternate_train
        self.save_idx = 0
        self.rollout_iteration = 0
        self.curiosity_model = CuriosityModel(
            observation_space, action_space, self.device, latent_dim, partially_observable, idm_net_arch, forward_net_arch
        )
        if model_path is not None and os.path.isfile(model_path):
            self.curiosity_model.load_state_dict(th.load(model_path))
        self.optimizer = th.optim.Adam(self.curiosity_model.parameters(), lr=learning_rate)
        self.forward_loss = nn.MSELoss()
        if isinstance(action_space, spaces.Box):
            self.inverse_dynamics_loss = nn.MSELoss()
        elif isinstance(action_space, spaces.Discrete):
            self.inverse_dynamics_loss = nn.CrossEntropyLoss()
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.inverse_dynamics_loss = MultiCrossEntropyLoss(action_space)
        else:
            raise NotImplementedError(f"{action_space} action space is not supported")

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        # Get the rollout buffer for curiosity reward calculation and training.
        rollout_buffer: EpisodicRolloutBuffer = self.locals["rollout_buffer"]
        # Given (s_t, a_t, s_{t+1}) calculate the curiosity reward and modify the original reward with it.
        self._calculate_curiosity_rewards(rollout_buffer)
        if not self.alternate_train or self.rollout_iteration % 2 == 0:
            # Training "forward model" and "inverse dynamics model".
            self._train_curiosity_model(rollout_buffer)
            if self.model_path is not None:
                # Save curiosity model.
                save_path = "{}{}.model".format(self.model_path, self.save_idx)
                self.save_idx += 1
                th.save(self.curiosity_model.state_dict(), save_path)
                logger.log(f"Saved the curiosity model at {save_path}.")
        # Soft reset the rollout buffer so that advantages can be calculated from rewards later.
        rollout_buffer.soft_reset()
        self.rollout_iteration += 1

    def _on_before_training(self) -> bool:
        # This callback will be called after the rollout end. Curiosity models will
        # be trained when the rollout iteration is even. Then, it will get incremented.
        # So, here if the rollout iteration is odd, we know that curiosity models were
        # trained and that the actor-critic model should not be trained, IF training is
        # to be alternated!
        if self.alternate_train and self.rollout_iteration % 2 == 1:
            dont_train = True
        else:
            dont_train = False
        return dont_train

    def _calculate_curiosity_rewards(self, rollout_buffer: EpisodicRolloutBuffer):
        # Initialize RNN-based dynamics model that extracts features for both "forward model" and "inverse dynamics model".
        self.curiosity_model.eval()
        self.curiosity_model.reset_hiddens(1)
        with th.no_grad():
            # Since the curiosity model assumes that we will give 1 episode for each call,
            # we must ensure each 'rollout_data' contains only 1 episode by setting the
            # minimum batch size to 1.
            for rollout_data in rollout_buffer.get(1):
                # assert rollout_data.dones[0].bool().item(), "At the first timestep of each episode, 'done' must be true!"
                if rollout_data.observations.shape[0] <= 1:
                    continue
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                elif isinstance(self.action_space, spaces.MultiDiscrete):
                    actions = actions.long()
                _, next_latent_features, latent_features = self.curiosity_model(
                    rollout_data.observations, actions, rollout_data.dones
                )
                curiosity_rewards = th.mean((next_latent_features - latent_features[1:]) ** 2, dim=1)
                if self.pure_curiosity_reward:
                    rollout_data.rewards[:-1] = self.curiosity_coefficient * curiosity_rewards
                else:
                    rollout_data.rewards[:-1] += self.curiosity_coefficient * curiosity_rewards
                # Write the calculated rewards back
                rollout_buffer.rewards[rollout_data.batch_inds.cpu().numpy()] = (
                    rollout_data.rewards.cpu().unsqueeze(-1).numpy()
                )

    def _train_curiosity_model(self, rollout_buffer: EpisodicRolloutBuffer):
        self.curiosity_model.train()
        for i in range(self.n_epochs):
            # Since the curiosity model assumes that we will give 1 episode for each call,
            # we must ensure each 'rollout_data' contains only 1 episode by setting the
            # minimum batch size to 1.
            for rollout_data in rollout_buffer.get(1):
                if rollout_data.observations.shape[0] <= 1:
                    continue
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                elif isinstance(self.action_space, spaces.MultiDiscrete):
                    actions = actions.long()
                self.optimizer.zero_grad()
                self.curiosity_model.reset_hiddens(1)
                first_n_minus_1_actions, next_latent_features, latent_features = self.curiosity_model(
                    rollout_data.observations, actions, rollout_data.dones
                )
                forward_loss = self.forward_loss(next_latent_features, latent_features[1:].detach())
                inverse_dynamics_loss = self.inverse_dynamics_loss(first_n_minus_1_actions, actions[:-1])
                loss: th.Tensor = forward_loss + inverse_dynamics_loss
                loss.backward()
                self.optimizer.step()
