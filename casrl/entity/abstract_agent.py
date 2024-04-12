import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from casrl.entity.position import Position
from casrl.enums.outcome import Outcome
from casrl.utils.const import GRID_HEIGHT, GRID_WIDTH


class AbstractAgent(ABC):

    @abstractmethod
    def __init__(
        self,
        size: int,
    ) -> None:
        self._position: None | Position = None
        self.size = size

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def run_iteration(self, *args: Any, **kwargs) -> tuple[Outcome, bool] | None:
        pass

    @property
    def position(self) -> Position:
        assert self._position is not None, "Position must be set before accessing position property"
        return self._position

    @position.setter
    def position(self, value: Position) -> None:
        self._position = value

    def distance_from(self, other: "AbstractAgent") -> float:
        return np.sqrt(
            np.power(
                self.position.x + int(self.size / 2) - (other.position.x + int(other.size / 2)), 2
            )
            + np.power(
                self.position.y + int(self.size / 2) - (other.position.y + int(other.size / 2)), 2
            )
        )

    def overlaps_with(self, other: "AbstractAgent") -> bool:
        if (
            self.position.x + self.size > other.position.x
            and self.position.x < other.position.x + other.size
            and self.position.y + self.size > other.position.y
            and self.position.y < other.position.y + other.size
        ):
            return True
        return False

    def is_out_of_bounds(self) -> bool:
        return (
            self.position.x <= 0 or self.position.x >= GRID_WIDTH - self.size or
            self.position.y <= 0 or self.position.y >= GRID_HEIGHT - self.size
        )

    def angle_from(self, other: "AbstractAgent") -> int:
        return int(math.degrees(math.atan2(
            (self.position.y + int(self.size / 2)) - (other.position.y + int(self.size / 2)),
            (self.position.x + int(self.size / 2)) - (other.position.x + int(self.size / 2)))
        ) + 180)


