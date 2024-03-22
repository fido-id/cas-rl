import os

import numpy as np

from casrl.const import GRID_WIDTH, GRID_HEIGHT
from casrl.entity.obstacle import Obstacle
from casrl.entity.position import Position


class Obstacles:
    def __init__(self, n_obstacles: int, obstacle_size: int):
        self.obstacles = [
            Obstacle(np.random.randint(0, GRID_WIDTH), int(GRID_HEIGHT / 3), obstacle_size)
            for _ in range(n_obstacles)
        ]
        self.obstacle_size = obstacle_size

    def run_iteration(self, player_position: Position):
        outcomes = []
        for obstacle in self.obstacles:
            outcomes.append(obstacle.run_iteration(player_position))

        return outcomes

    def reset(self):
        for obstacle in self.obstacles:
            obstacle.reset(
                np.random.randint(1, GRID_WIDTH - self.obstacle_size),
                np.random.randint(1, GRID_HEIGHT - self.obstacle_size)
            )

    def save_qtables(self, root_path):
        os.makedirs(root_path, exist_ok=True)
        for i, obstacle in enumerate(self.obstacles):
            obstacle_state_path = f"{root_path}/{i}.npy"
            obstacle.save_qtable(obstacle_state_path)
            print(f"Saved obstacle {i} state to {obstacle_state_path}")

    def load_qtables(self, root_path):
        if not os.path.exists(root_path):
            print(f"Cannot load RL state from {root_path}")
            return

        for i, obstacle in enumerate(self.obstacles):
            obstacle.load_qtable(f"{root_path}/{i}.npy")

        print(f"Loaded RL states from {root_path}")
