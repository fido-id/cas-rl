import numpy as np
import pygame

from casrl.const import MOVEMENT_OFFSET, GRID_HEIGHT, GRID_WIDTH, AGENT_SIZE
from casrl.entity.agent.abstract_agent import AbstractAgent
from casrl.entity.position import Position

OFFSET = 1


class PlayableAgent(AbstractAgent):
    def __init__(self, size: int):
        self.size = size
        self._position = None
        self.reset()

    def run_iteration(self, keys):
        if keys[pygame.K_w]:
            self.position.y = max(self.position.y - MOVEMENT_OFFSET, 0)
        if keys[pygame.K_s]:
            self.position.y = min(self.position.y + MOVEMENT_OFFSET, GRID_HEIGHT - self.size)
        if keys[pygame.K_a]:
            self.position.x = max(self.position.x - MOVEMENT_OFFSET, 0)
        if keys[pygame.K_d]:
            self.position.x = min(self.position.x + MOVEMENT_OFFSET, GRID_WIDTH - self.size)

    def reset(self):
        self._position = Position(
            np.random.randint(1, GRID_WIDTH - self.size),
            np.random.randint(1, GRID_HEIGHT - self.size),
            self.size
        )

    def reset_to_fixed_pos(self) -> None:
        self._position = Position(
            np.random.randint(4, GRID_WIDTH - self.size - 4),
            GRID_HEIGHT - 3,
            self.size
        )

    @property
    def position(self) -> Position:
        return self._position
