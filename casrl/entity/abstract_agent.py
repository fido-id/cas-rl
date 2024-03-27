from abc import ABC, abstractmethod

from casrl.entity.position import Position
from casrl.enums.outcome import Outcome


class AbstractAgent(ABC):

    @abstractmethod
    def __init__(
        self,
        size: int,
    ):
        self._position = None
        self.size = size

    @abstractmethod
    def reset(self, **kwargs) -> None:
        pass

    @property
    def position(self) -> Position:
        return self._position

    @abstractmethod
    def run_iteration(self, **kwargs) -> tuple[Outcome, bool]:
        pass

