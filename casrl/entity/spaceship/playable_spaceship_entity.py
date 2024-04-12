import numpy as np
import pygame

from casrl.entity.abstract_agent import AbstractAgent
from casrl.entity.position import Position
from casrl.utils.const import GRID_HEIGHT, GRID_WIDTH, MOVEMENT_OFFSET


class PlayableSpaceship(AbstractAgent):
    def __init__(self, size: int) -> None:
        self.size = size
        self._position: None | Position = None
        self.reset()

    def run_iteration(self, keys: pygame.key.ScancodeWrapper) -> None:
        if keys[pygame.K_w]:
            self.position.y = max(self.position.y - MOVEMENT_OFFSET, 0)
        if keys[pygame.K_s]:
            self.position.y = min(self.position.y + MOVEMENT_OFFSET, GRID_HEIGHT - self.size)
        if keys[pygame.K_a]:
            self.position.x = max(self.position.x - MOVEMENT_OFFSET, 0)
        if keys[pygame.K_d]:
            self.position.x = min(self.position.x + MOVEMENT_OFFSET, GRID_WIDTH - self.size)

    def reset(self) -> None:
        self._position = Position(
            np.random.randint(1, GRID_WIDTH - self.size),
            np.random.randint(1, GRID_HEIGHT - self.size),
        )

    def reset_to_fixed_pos(self) -> None:
        self._position = Position(
            np.random.randint(4, GRID_WIDTH - self.size - 4),
            GRID_HEIGHT - 3,
        )
