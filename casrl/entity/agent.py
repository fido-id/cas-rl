import pygame

from casrl.const import MOVEMENT_OFFSET, GRID_HEIGHT, GRID_WIDTH, AGENT_SIZE
from casrl.entity.position import Position

OFFSET = 1


class Agent:
    def __init__(self, initial_x: int, initial_y: int, size: int):
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.size = size
        self.position = Position(initial_x, initial_y, size)
        self.size = size

    def run_episode(self, keys):
        if keys[pygame.K_w]:
            self.position.y = max(self.position.y - MOVEMENT_OFFSET, 0)
        if keys[pygame.K_s]:
            self.position.y = min(self.position.y + MOVEMENT_OFFSET, GRID_HEIGHT - self.size)
        if keys[pygame.K_a]:
            self.position.x = max(self.position.x - MOVEMENT_OFFSET, 0)
        if keys[pygame.K_d]:
            self.position.x = min(self.position.x + MOVEMENT_OFFSET, GRID_WIDTH - self.size)

    def reset(self):
        self.position = Position(self.initial_x, self.initial_y, self.size)
