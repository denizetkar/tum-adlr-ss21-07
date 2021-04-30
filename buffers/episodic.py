from stable_baselines3.common.buffers import RolloutBuffer

# TODO:
# + Minibatches over episodes instead of over individual experiences (for RNN),
#   * Batch size will still be in number of timesteps instead of number of episodes so that
#     each minibatch contains more or less equal experience.
#   * Modify RolloutBuffer.get() method for implementation.
#   * Pass "_init_setup_model=False" as parameter to PPO.__init__() method and do the initialization
#     yourself (self._setup_model()) so that you can use your own rollout buffer.
#   * Make sure to also return "dones" in minibatches to distinguish episode borders during
#     RNN training -> reimplement RolloutBufferSamples class.


class EpisodicRolloutBuffer(RolloutBuffer):
    pass
