import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main(args: argparse.Namespace):
    # Parallel environments
    env = make_vec_env("CartPole-v1", n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_cartpole")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_cartpole")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add all needed arguments.
    main(parser.parse_args())
