from stable_baselines3.common import callbacks

# TODO:
# + callback.on_rollout_start() method must implement,
#   * Initializing RNN-based dynamics model that extracts features for both "forward model" and "inverse dynamics model".
#   * Note: It is the one called in OnPolicyAlgorithm.collect_rollouts() method.

# + callback.on_step() method must implement,
#   * Given (s_t, a_t, s_{t+1}) calculate the curiosity reward and modify the original reward with it.
#   * Note: It is the one called in OnPolicyAlgorithm.collect_rollouts() method.

# + callback.on_rollout_end() method must implement,
#   * Training "forward model" and "inverse dynamics model".
#   * Note: It is the one called in OnPolicyAlgorithm.collect_rollouts() method.


class CuriosityCallback(callbacks.BaseCallback):
    pass
