from typing import List, Optional, Tuple

import gym
import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import is_image_space, preprocess_obs
from stable_baselines3.common.torch_layers import FlattenExtractor, NatureCNN

from .extractor import RnnExtractor
from .modules import MLP, Reshape, SequentialExpand, TupleApply, TuplePick

# See the blog below for a good explanation:
# https://medium.com/analytics-vidhya/advanced-exploration-curiosity-driven-exploration-52bcac6d3450


class InverseDynamicsModel(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: th.device,
        latent_dim: int = 64,
        partially_observable: bool = True,
        net_arch: Optional[List[int]] = None,
    ):
        super().__init__()
        self.observation_space = observation_space
        # Get first extractor.
        if is_image_space(observation_space):
            self.first_extractor = NatureCNN(observation_space, latent_dim)
        else:
            flattener = FlattenExtractor(observation_space)
            self.first_extractor = nn.Sequential(flattener, nn.Linear(flattener.features_dim, latent_dim)).to(device)
        # Get second extractor.
        if net_arch is None:
            net_arch = []
        net_arch.append(latent_dim)
        self.rnn_extractor: Optional[RnnExtractor] = None
        if partially_observable:
            self.rnn_extractor = RnnExtractor(latent_dim, net_arch=net_arch, device=device).to(device)
            self.second_extractor = SequentialExpand(
                TupleApply(Reshape(-1, 1, latent_dim), Reshape(-1, 1)),
                self.rnn_extractor,
                TupleApply(Reshape(-1, latent_dim), nn.Identity()),
                TuplePick(0),
            ).to(device)
        else:
            self.second_extractor = SequentialExpand(
                TuplePick(0), MLP(latent_dim, net_arch=net_arch, activation_fn=nn.PReLU)
            ).to(device)
        # Get action network.
        action_dist = make_proba_distribution(action_space)
        if isinstance(action_dist, DiagGaussianDistribution):
            self.action_net, _ = action_dist.proba_distribution_net(
                latent_dim=latent_dim * 2,
            ).to(device)
        elif isinstance(action_dist, CategoricalDistribution):
            self.action_net = action_dist.proba_distribution_net(latent_dim=latent_dim * 2).to(device)
        elif isinstance(action_dist, MultiCategoricalDistribution):
            self.action_net = action_dist.proba_distribution_net(latent_dim=latent_dim * 2).to(device)
        else:
            raise NotImplementedError(f"Unsupported distribution '{action_dist}'.")

    def forward(self, obs: th.Tensor, dones: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # 'obs' is of shape (seq_len,) + self.obs_shape
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape(-1)
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=True)
        first_features = self.first_extractor(preprocessed_obs)
        latent_features = self.second_extractor(first_features, dones)
        # Assume that there is only 1 episode in the 'obs' !!!!!!!!!!!!!!!!!!!!!
        current_and_next_latents = th.cat([latent_features[:-1], latent_features[1:]], dim=-1)
        first_n_minus_1_actions = self.action_net(current_and_next_latents)
        return first_n_minus_1_actions, latent_features

    def reset_hiddens(self, batch_size: int = 1):
        if self.rnn_extractor is not None:
            self.rnn_extractor.reset_hiddens(batch_size)


class ForwardModel(nn.Module):
    def __init__(
        self,
        action_space: gym.Space,
        device: th.device,
        latent_dim: int = 64,
        net_arch: Optional[List[int]] = None,
    ):
        super().__init__()
        self.action_space = action_space
        if net_arch is None:
            net_arch = []
        net_arch.append(latent_dim)
        self.flattener = FlattenExtractor(action_space)
        self.net = MLP(latent_dim + self.flattener.features_dim, net_arch).to(device)

    def forward(self, latent_features: th.Tensor, actions: th.Tensor) -> th.Tensor:
        preprocessed_actions = preprocess_obs(actions, self.action_space, normalize_images=False)
        preprocessed_actions = self.flattener(preprocessed_actions)
        x = th.cat([latent_features[:-1], preprocessed_actions[:-1]], dim=-1)
        next_latent_features = self.net(x)
        return next_latent_features


class CuriosityModel(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: th.device,
        latent_dim: int = 64,
        partially_observable: bool = True,
        idm_net_arch: Optional[List[int]] = None,
        forward_net_arch: Optional[List[int]] = None,
    ):
        super().__init__()
        self.inverse_dynamics_model = InverseDynamicsModel(
            observation_space, action_space, device, latent_dim, partially_observable, idm_net_arch
        )
        self.forward_model = ForwardModel(action_space, device, latent_dim, forward_net_arch)

    def forward(self, obs: th.Tensor, actions: th.Tensor, dones: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        first_n_minus_1_actions, latent_features = self.inverse_dynamics_model(obs, dones)
        next_latent_features = self.forward_model(latent_features.detach(), actions)
        return first_n_minus_1_actions, next_latent_features, latent_features

    def reset_hiddens(self, batch_size: int = 1):
        self.inverse_dynamics_model.reset_hiddens(batch_size=batch_size)
