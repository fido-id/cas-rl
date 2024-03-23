from abc import ABC, abstractmethod

from casrl.entity.obstacles import Obstacles
from casrl.entity.position import Position


class AbstractAgent(ABC):
    @abstractmethod
    def run_iteration(self, npcs: Obstacles) -> int:
        pass

    @abstractmethod
    def reset_to_fixed_pos(self) -> None:
        pass

    @property
    @abstractmethod
    def position(self) -> Position:
        pass


