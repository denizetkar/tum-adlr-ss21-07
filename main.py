import argparse
import os

import numpy as np
import torch as th
from stable_baselines3.common.env_util import make_vec_env

from callbacks import curiosity
from rl_algo import RecurrentPPO

# from utils.profiler import profile
th.set_default_dtype(th.float32)


# @profile(file_path="profile.pstats")
def main(args: argparse.Namespace):
    env = make_vec_env("CartPole-v1", n_envs=4)
    if args.ppo_model_path is not None and os.path.isfile(args.ppo_model_path):
        model = RecurrentPPO.load(args.ppo_model_path, env=env, device=args.device)
    else:
        policy_kwargs = {"net_arch": [16, dict(pi=[16], vf=[16])]}
        model = RecurrentPPO(
            "RnnPolicy", env, n_steps=256, min_batch_size=64, policy_kwargs=policy_kwargs, device=args.device, verbose=1
        )

    model.learn(
        total_timesteps=20000,
        callback=curiosity.CuriosityCallback(
            env.observation_space,
            env.action_space,
            partially_observable=False,
            idm_net_arch=[16],
            forward_net_arch=[16],
            model_path=args.curiosity_model_path,
            device=args.device,
        ),
    )

    if args.ppo_model_path is not None:
        model.save(args.ppo_model_path)

    obs = env.reset()
    dones = np.zeros((env.num_envs,), dtype=bool)
    while True:
        action = model.predict(obs, dones)
        obs, _, dones, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curiosity-model-path", type=str, help="Path to the curiosity model file to be loaded/saved.")
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
    main(parser.parse_args())
