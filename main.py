import argparse

import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from rl_algo import RecurrentPPO

# from utils.profiler import profile


# @profile(file_path="profile.pstats")
def main(args: argparse.Namespace):
    env = make_vec_env("CartPole-v1", n_envs=4)

    policy_kwargs = {"net_arch": [1024, dict(pi=[1024], vf=[1024])]}
    model = RecurrentPPO("RnnPolicy", env, n_steps=256, min_batch_size=64, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=1025)

    # model.save("ppo_cartpole")
    # del model  # remove to demonstrate saving and loading
    # model = PPO.load("ppo_cartpole")

    obs = env.reset()
    dones = np.zeros((env.num_envs,), dtype=bool)
    while True:
        action = model.predict(obs, dones)
        obs, _, dones, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add all needed arguments.
    main(parser.parse_args())
