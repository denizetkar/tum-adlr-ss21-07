import argparse
import os

import torch as th
from stable_baselines3.common.env_util import make_vec_env

from callbacks import curiosity
from rl_algo import RecurrentPPO

# from utils.profiler import profile
th.set_default_dtype(th.float32)


# @profile(file_path="profile.pstats")
def main(args: argparse.Namespace):
    # rnn_hidden_dim
    env = make_vec_env(args.env, n_envs=args.n_envs)
    if args.ppo_model_path is not None and os.path.isfile(args.ppo_model_path):
        model = RecurrentPPO.load(args.ppo_model_path, env=env, device=args.device)
    else:
        policy_kwargs = {"net_arch": [512, dict(pi=[512], vf=[512])]}
        model = RecurrentPPO(
            args.policy,
            env,
            n_steps=256,
            min_batch_size=64,
            policy_kwargs=policy_kwargs,
            device=args.device,
            verbose=1,
            tensorboard_log=args.tensorboard_log
        )
    callback = curiosity.CuriosityCallback(
        model.env.observation_space,
        model.env.action_space,
        partially_observable=False,
        idm_net_arch=[512],
        forward_net_arch=[512],
        model_path=args.curiosity_model_path,
        device=args.device,
    ) if args.use_curiosity else None

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
    )

    if args.ppo_model_path is not None:
        model.save(args.ppo_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-curiosity", action='store_true', help="Flag for using curiosity in the training")
    parser.add_argument(
        "--curiosity-model-path",
        type=str,
        required=True,
        help="Path to the curiosity model file to be loaded/saved.")
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        help="Path to the `RecurrentPPO` model file to be loaded/saved. Note that it is a '.zip' file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="String representation of the device to be used by PyTorch."
        "See https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device for more details.",
    )
    parser.add_argument("--total-timesteps", type=int, default=20000, help="Total number of timestamps for training")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of environments for data collection")
    parser.add_argument(
        "--policy",
        type=str,
        default="CnnRnnPolicy",
        choices=["RnnPolicy", "CnnRnnPolicy"],
        help="Type of the policy network")
    parser.add_argument(
        "--env",
        type=str,
        default="BreakoutNoFrameskip-v4",
        help="String representation of the gym environment"
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        required=True,
        help="String representation of the gym environment"
    )
    main(parser.parse_args())
