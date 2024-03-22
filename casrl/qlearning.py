import math

import numpy as np

from casrl.action import Action
from casrl.entity.position import Position
from casrl.const import MOVEMENT_OFFSET, GRID_WIDTH, GRID_HEIGHT, CAS_THRESHOLD


class QLearning:

    def __init__(
        self,
        n_possible_actions: int,
        exploration_rate: float = 0.3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9
    ):
        self.qtable = np.zeros((360, math.ceil((CAS_THRESHOLD**2 + CAS_THRESHOLD**2)**(1/2)), n_possible_actions),
                               dtype=np.float32)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def draw_action(self, obstacle_position: Position, player_position: Position) -> int:
        angle = obstacle_position.angle_from(player_position)
        if angle == 360:
            angle = 0

        distance = int(obstacle_position.distance(player_position))
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(range(len(Action)))
        else:
            return np.argmax(
                self.qtable[angle, distance]
            )

    def update_qtable(self, obstacle_position, player_position, action, reward, new_obstacle_position, is_terminal_state):
        angle = obstacle_position.angle_from(player_position)
        new_angle = new_obstacle_position.angle_from(player_position)
        if angle == 360:
            angle = 0
        if new_angle == 360:
            new_angle = 0

        distance = int(obstacle_position.distance(player_position))
        new_distance = int(new_obstacle_position.distance(player_position))
        max_expected_reward = self.discount_factor * np.max(self.qtable[new_angle, new_distance])
        if is_terminal_state:
            max_expected_reward = 0
        self.qtable[angle, distance, action] += self.learning_rate * (reward + max_expected_reward - self.qtable[angle, distance, action])

    def store_qtable(self, path: str):
        np.save(path, self.qtable)

    def load_qtable(self, path: str):
        self.qtable = np.load(path)
