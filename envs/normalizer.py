import gym

# TODO: Implement normalized environment for observation normalization.


class NormalizedEnv:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
