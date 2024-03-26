from abc import ABC, abstractmethod

from casrl.entity.position import Position


class AbstractSpaceship(ABC):
    @abstractmethod
    def run_iteration(self, **kwargs) -> int:
        pass

    @abstractmethod
    def reset_to_fixed_pos(self) -> None:
        pass

    @property
    @abstractmethod
    def position(self) -> Position:
        pass


