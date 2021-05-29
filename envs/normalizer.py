import gym

# TODO: Implement normalized environment for observation normalization.
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
# Note: Do not use normalizing environment with image inputs!!!!


class NormalizedEnv:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
