import os

import numpy as np

from casrl.const import GRID_WIDTH, GRID_HEIGHT
from casrl.entity.obstacle import Obstacle
from casrl.entity.position import Position
from casrl.entity.reward import Reward


class Obstacles:
    def __init__(self, n_obstacles: int, obstacle_size: int, reward_function: Reward):

        self.obstacles = [
            Obstacle(obstacle_size, reward_function)
            for _ in range(n_obstacles)
        ]
        self.obstacle_size = obstacle_size

    def run_iteration(self, player_position: Position):
        outcomes = []
        for obstacle in self.obstacles:
            outcomes.append(obstacle.run_iteration(player_position))

        return outcomes

    def reset(self, agent_position: Position):
        for obstacle in self.obstacles:
            obstacle.reset(agent_position)

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
