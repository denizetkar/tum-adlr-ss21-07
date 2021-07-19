from typing import List, Union

from stable_baselines3.common import callbacks


class EnhancedBaseCallback(callbacks.BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_before_training(self) -> bool:
        return self._on_before_training()

    def _on_before_training(self) -> bool:
        return False


MaybeCallback = Union[None, List[EnhancedBaseCallback], EnhancedBaseCallback]
