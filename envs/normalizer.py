import gym

# TODO: Implement normalized environment for observation normalization.
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize


class NormalizedEnv:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
