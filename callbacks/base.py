from typing import Callable, List, Union

from stable_baselines3.common import callbacks


class EnhancedBaseCallback(callbacks.BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def on_before_training(self) -> bool:
        return self._on_before_training()

    def _on_before_training(self) -> bool:
        raise NotImplementedError


MaybeCallback = Union[None, Callable, List[EnhancedBaseCallback], EnhancedBaseCallback]
