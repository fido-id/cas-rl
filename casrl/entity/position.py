import math

import numpy as np

from casrl.utils.const import GRID_WIDTH, GRID_HEIGHT


class Position:
    def __init__(self, x: int, y: int, size: int) -> None:
        self.x = x
        self.y = y
        self.size = size

    def update_with_ooo_check(self, delta_x: int, delta_y: int) -> None:
        if delta_x > 0:
            self.x = min(self.x + delta_x, GRID_WIDTH - 1)
        elif delta_x < 0:
            self.x = max(self.x + delta_x, 0)
        if delta_y > 0:
            self.y = min(self.y + delta_y, GRID_HEIGHT - 1)
        elif delta_y < 0:
            self.y = max(self.y + delta_y, 0)

    def update(self, delta_x: int, delta_y: int) -> None:
        self.x = self.x + delta_x
        self.y = self.y + delta_y

    def distance_from(self, other: 'Position') -> float:
        return np.sqrt(
            np.power(
                self.x + int(self.size / 2) - (other.x + int(other.size / 2)), 2
            )
            + np.power(
                self.y + int(self.size / 2) - (other.y + int(other.size / 2)), 2
            )
        )

    def overlaps_with(self, other: 'Position') -> bool:
        if self.x + self.size >= other.x and self.x <= other.x + other.size and \
                self.y + self.size >= other.y and self.y <= other.y + other.size:
            return True
        return False

    def is_out_of_bounds(self) -> bool:
        return (
            self.x <= 0 or self.x >= GRID_WIDTH - self.size or self.y <= 0 or self.y >= GRID_HEIGHT - self.size
        )

    def angle_from(self, other: 'Position') -> int:
        return int(math.degrees(math.atan2(
            (self.y + int(self.size / 2)) - (other.y + int(self.size / 2)),
            (self.x + int(self.size / 2)) - (other.x + int(self.size / 2)))
        ) + 180)
