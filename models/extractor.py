from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from efficientnet_pytorch import EfficientNet
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from torch import nn
from torchvision import models

from .mlp import MLP
from .rnn.lstms import LayerNormSemeniutaLSTM, MultiLayerLSTM


class RnnExtractor(nn.Module):
    """
    Constructs an RNN that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module] = nn.PReLU,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        shared_layers: List[int] = []
        policy_layers: List[int] = []  # Layer sizes of the network that only belongs to the policy network
        value_layers: List[int] = []  # Layer sizes of the network that only belongs to the value network

        # Iterate through the shared layers and build the shared parts of the network
        last_shared_dim = feature_dim
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                shared_layers.append(layer)
                last_shared_dim = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_layers = layer["pi"]
                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        # Save dim, used to create the distributions
        self.latent_dim_pi = policy_layers[-1] if len(policy_layers) > 0 else last_shared_dim
        self.latent_dim_vf = value_layers[-1] if len(value_layers) > 0 else last_shared_dim

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = MultiLayerLSTM(feature_dim, LayerNormSemeniutaLSTM, shared_layers, dropout=0.0).to(device)
        self.policy_net = MultiLayerLSTM(last_shared_dim, LayerNormSemeniutaLSTM, policy_layers, dropout=0.0).to(device)
        self.value_net = MultiLayerLSTM(last_shared_dim, LayerNormSemeniutaLSTM, value_layers, dropout=0.0).to(device)

    def reset_parameters(self):
        self.shared_net.reset_parameters()
        self.policy_net.reset_parameters()
        self.value_net.reset_parameters()

    def reset_hiddens(self, batch_size: int = 1):
        self.shared_net.reset_hiddens(batch_size=batch_size)
        self.policy_net.reset_hiddens(batch_size=batch_size)
        self.value_net.reset_hiddens(batch_size=batch_size)

    def reset_batch_hiddens(self, batch_idx: int):
        self.shared_net.reset_batch_hiddens(batch_idx=batch_idx)
        self.policy_net.reset_batch_hiddens(batch_idx=batch_idx)
        self.value_net.reset_batch_hiddens(batch_idx=batch_idx)

    def forward(self, features: th.Tensor, dones: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :param features: feature tensor of shape (seq_len, n_envs, feature_dim)
        :param dones: Indicator for a new episode start of shape (seq_len, n_envs)
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        seq_len = features.size(0)
        policy_outputs: List[th.Tensor] = []
        value_outputs: List[th.Tensor] = []
        for i in range(seq_len):
            done_indexes: List[int] = dones[i].flatten().nonzero().flatten().cpu().tolist()
            for done_idx in done_indexes:
                self.reset_batch_hiddens(done_idx)
            shared_latent = self.shared_net(features[i].unsqueeze(0))
            policy_output, value_output = self.policy_net(shared_latent), self.value_net(shared_latent)
            policy_outputs.append(policy_output)
            value_outputs.append(value_output)

        return th.cat(policy_outputs, dim=0), th.cat(value_outputs, dim=0)


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module] = nn.PReLU,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        shared_layers: List[int] = []
        policy_layers: List[int] = []  # Layer sizes of the network that only belongs to the policy network
        value_layers: List[int] = []  # Layer sizes of the network that only belongs to the value network

        # Iterate through the shared layers and build the shared parts of the network
        last_shared_dim = feature_dim
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                shared_layers.append(layer)
                last_shared_dim = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_layers = layer["pi"]
                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        # Save dim, used to create the distributions
        self.latent_dim_pi = policy_layers[-1] if len(policy_layers) > 0 else last_shared_dim
        self.latent_dim_vf = value_layers[-1] if len(value_layers) > 0 else last_shared_dim

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = MLP(feature_dim, shared_layers, activation_fn=activation_fn, dropout=0.0).to(device)
        self.policy_net = MLP(last_shared_dim, policy_layers, activation_fn=activation_fn, dropout=0.0).to(device)
        self.value_net = MLP(last_shared_dim, value_layers, activation_fn=activation_fn, dropout=0.0).to(device)

    def reset_parameters(self):
        return

    def reset_hiddens(self, batch_size: int = 1):
        return

    def reset_batch_hiddens(self, batch_idx: int):
        return

    def forward(self, features: th.Tensor, dones: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :param features: feature tensor of shape (seq_len, n_envs, feature_dim)
        :param dones: Indicator for a new episode start of shape (seq_len, n_envs)
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


class CnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, device: Union[th.device, str] = "auto", features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        device = get_device(device)
        assert is_image_space(observation_space), (
            "You should use CnnExtractor "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnRnnPolicy` instead of `RnnPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        assert n_input_channels == 3, "Number of input channels should be 3 for images"
        pretrained_vgg = models.vgg16(pretrained=True)
        self.features = pretrained_vgg.features.to(device)

        with th.no_grad():
            latent_dims = self.features(th.as_tensor(observation_space.sample()[None]).to(device).float()).shape

        latent_h, latent_w = latent_dims[2:]
        self.avg_pool = nn.AdaptiveAvgPool2d((latent_h, latent_w)).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(512 * latent_h * latent_w, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.features_dim),
        ).to(device)
        # TODO: Initialize weights to use this extractor

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = th.flatten(x, 1)
        x = self.classifier(x)
        return x


class EfficientNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, device: Union[th.device, str] = "auto", features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        device = get_device(device)
        assert is_image_space(observation_space), (
            "You should use CnnExtractor "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnRnnPolicy` instead of `RnnPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        assert n_input_channels == 3, "Number of input channels should be 3 for images"
        self.efficient_net = EfficientNet.from_pretrained("efficientnet-b1", num_classes=features_dim).to(device)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.efficient_net(x)
        return x
