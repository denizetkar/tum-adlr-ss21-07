import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import animation
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

from callbacks import curiosity
from rl_algo import RecurrentPPO

# from utils.profiler import profile
th.set_default_dtype(th.float32)


# @profile(file_path="profile.pstats")
def train(args: argparse.Namespace):
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    th.manual_seed(args.rng_seed)

    if args.atari:
        env = VecFrameStack(venv=make_atari_env(args.env, n_envs=args.n_envs, seed=args.rng_seed), n_stack=4)
    else:
        env = make_vec_env(args.env, n_envs=args.n_envs, seed=args.rng_seed)

    if args.ppo_model_path is not None and os.path.isfile(args.ppo_model_path):
        model = RecurrentPPO.load(args.ppo_model_path, env=env, device=args.device)
    else:
        if args.max_absolute_reward is None and args.atari:
            args.max_absolute_reward = 1.0
        if args.use_curiosity:
            if args.pure_curiosity_reward:
                args.max_absolute_reward = args.curiosity_coefficient
            elif args.max_absolute_reward is not None:
                args.max_absolute_reward += args.curiosity_coefficient
        policy_kwargs = {
            "net_arch": [
                args.rnn_hidden_dim,
                args.rnn_hidden_dim,
                dict(pi=[args.rnn_hidden_dim, args.rnn_hidden_dim], vf=[args.rnn_hidden_dim, args.rnn_hidden_dim]),
            ]
        }
        model = RecurrentPPO(
            args.policy,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            min_batch_size=args.min_batch_size,
            n_epochs=args.ppo_epochs,
            max_absolute_reward=args.max_absolute_reward,
            max_eps_len=args.max_eps_len,
            policy_kwargs=policy_kwargs,
            device=args.device,
            model_path=args.ppo_model_path,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
        )
    callback = (
        curiosity.CuriosityCallback(
            model.env.observation_space,
            model.env.action_space,
            learning_rate=args.learning_rate,
            n_epochs=args.curiosity_epochs,
            curiosity_coefficient=args.curiosity_coefficient,
            reg_coefficient=args.curiosity_reg_coef,
            latent_dim=args.rnn_hidden_dim,
            partially_observable=args.partially_observable,
            pure_curiosity_reward=args.pure_curiosity_reward,
            idm_net_arch=[args.rnn_hidden_dim, args.rnn_hidden_dim],
            forward_net_arch=[args.rnn_hidden_dim, args.rnn_hidden_dim],
            model_path=args.curiosity_model_path,
            device=args.device,
            alternate_train=args.alternate_train,
        )
        if args.use_curiosity
        else None
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
    )


def play(args: argparse.Namespace):
    if args.pybullet_env:
        import pybullet_envs

    if args.atari:
        env = VecFrameStack(venv=make_atari_env(args.env, n_envs=args.n_envs), n_stack=4)
    else:
        env = make_vec_env(args.env, n_envs=args.n_envs)

    model: RecurrentPPO = RecurrentPPO.load(args.ppo_model_path, env=env, device=args.device)

    if args.pybullet_env:
        env.render()
    obs = env.reset()
    dones = np.zeros((env.num_envs,), dtype=bool)
    model.reset_hiddens()
    frames = []
    while True:
        time.sleep(0.005)
        action = model.predict(obs, dones)
        obs, _, dones, _ = env.step(action)
        if args.save_as_gif:
            frames.append(env.render(mode="rgb_array"))
            if len(frames) == args.gif_frame_size:
                break
        elif not args.pybullet_env:
            env.render()
    env.close()
    if args.save_as_gif:
        save_frames_as_gif(frames, args.gif_path)


def save_frames_as_gif(frames, gif_path="gym_animation.gif"):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(gif_path, writer="imagemagick", fps=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title="Subcommand")
    train_parser = subparser.add_parser("train", help="Subcommand for training the PPO and curiosity models.")
    train_parser.add_argument(
        "--curiosity-model-path",
        type=str,
        help="Path to the curiosity model file to be loaded/saved. Please omit an extension.",
    )
    train_parser.add_argument(
        "--ppo-model-path",
        type=str,
        required=True,
        help="Path to the `RecurrentPPO` model file to be loaded/saved. Please omit an extension.",
    )
    train_parser.add_argument(
        "--tensorboard-log",
        type=str,
        required=True,
        help="Path to the directory for saving the tensorboard logs. Directory will be created if it does not exist.",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="String representation of the device to be used by PyTorch."
        "See https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device for more details.",
    )
    train_parser.add_argument(
        "--alternate-train",
        action="store_true",
        help="Flag for making alternating training between curiosity models and PPO. Only relevant when curiosity is used!",
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate to be used for all types of training."
    )
    train_parser.add_argument(
        "--min-batch-size",
        type=int,
        default=64,
        help="Minimum batch size for training PPO. Must be >0 and <=(n_envs * n_steps)",
    )
    train_parser.add_argument("--ppo-epochs", type=int, default=10, help="Number of training epochs for PPO.")
    train_parser.add_argument(
        "--curiosity-epochs", type=int, default=3, help="Number of epochs to train the curiosity models per 'collect_rollout'."
    )
    train_parser.add_argument(
        "--curiosity-coefficient",
        type=float,
        default=0.5,
        help="The coefficient to multiply the curiosity reward before adding to the total rewards.",
    )
    train_parser.add_argument(
        "--curiosity-reg-coef", type=float, default=0.001, help="Regularization coefficient for training curiosity models."
    )
    train_parser.add_argument("--total-timesteps", type=int, default=200000, help="Total number of timestamps for training.")
    train_parser.add_argument(
        "--n-steps", type=int, default=1024, help="Maximum number of timesteps per rollout per environment."
    )
    train_parser.add_argument("--n-envs", type=int, default=16, help="Number of environments for data collection.")
    train_parser.add_argument("--rnn-hidden-dim", type=int, default=512, help="Hidden dimension size for RNNs.")
    train_parser.add_argument(
        "--policy",
        type=str,
        default="MlpPolicy",
        choices=["MlpPolicy", "CnnPolicy", "RnnPolicy", "CnnRnnPolicy"],
        help="Type of the policy network.",
    )
    train_parser.add_argument(
        "--env", type=str, default="MountainCarContinuous-v0", help="String representation of the gym environment."
    )
    train_parser.add_argument(
        "--max-absolute-reward", type=float, help="Maximum value of the absolute reward from the environment."
    )
    train_parser.add_argument("--max-eps-len", type=int, help="Maximum possible length of an episode in the environment.")
    train_parser.add_argument("--atari", action="store_true", help="Flag for performing Atari preprocessing on observations.")
    train_parser.add_argument(
        "--rng-seed",
        type=int,
        default=12345,
        help="The seed to be used for all random number generators for repeatable experiments.",
    )
    train_parser.add_argument(
        "--partially-observable",
        action="store_true",
        help="Flag for informing the model to use RNNs due to partial observability of the RL problem.",
    )
    train_parser.add_argument("--use-curiosity", action="store_true", help="Flag for using curiosity in the training.")
    train_parser.add_argument(
        "--pure-curiosity-reward",
        action="store_true",
        help="Flag for telling to use pure curiosity rewards instead of "
        "mixed curiosity reward (extrinsic + coef * curiosity).",
    )
    train_parser.set_defaults(func=train)

    play_parser = subparser.add_parser(
        "play", help="Subcommand for getting the PPO model to play in the environment for which it was trained."
    )
    play_parser.add_argument(
        "--ppo-model-path",
        type=str,
        required=True,
        help="Path to the `RecurrentPPO` model file to be loaded/saved. Note that it is a '.zip' file.",
    )
    play_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="String representation of the device to be used by PyTorch."
        "See https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device for more details.",
    )
    play_parser.add_argument("--n-envs", type=int, default=4, help="Number of environments for data collection.")
    play_parser.add_argument(
        "--env", type=str, default="MountainCarContinuous-v0", help="String representation of the gym environment."
    )
    play_parser.add_argument(
        "--pybullet-env",
        action="store_true",
        help="Flag for specifying if the environment is that of PyBullet and must be rendered accordingly.",
    )
    play_parser.add_argument("--atari", action="store_true", help="Flag for performing Atari preprocessing on observations.")
    play_parser.add_argument("--save-as-gif", action="store_true", help="Flag for saving the played episodes as a gif.")
    play_parser.add_argument("--gif-frame-size", type=int, default=1000, help="Number of frames to be saved as a gif.")
    play_parser.add_argument("--gif-path", type=str, default="gym_animation.gif", help="Path to save the .gif file.")
    play_parser.set_defaults(func=play)

    args = parser.parse_args()
    args.func(args)
