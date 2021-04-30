from stable_baselines3.common import callbacks

# TODO:
# + callback.on_rollout_end() method must implement,
#   * Schedule custom dynamic hyperparameters (standard dynamic hyperparameters are scheduled in PPO.train() method).
#   * Note: It is the one called in OnPolicyAlgorithm.collect_rollouts() method.


class HyperparameterCallback(callbacks.BaseCallback):
    pass
