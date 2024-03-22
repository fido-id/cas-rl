import numpy as np

from casrl.action import Action
from casrl.entity.position import Position
from casrl.const import MOVEMENT_OFFSET, GRID_WIDTH, GRID_HEIGHT


class QLearning:

    def __init__(
        self,
        n_possible_actions: int,
        exploration_rate: float = 0.1,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9
    ):
        # one couple grid_width, grid_height is for the obstacle and the other for the player
        self.qtable = np.zeros((GRID_WIDTH + 2, GRID_HEIGHT + 2, GRID_WIDTH + 2, GRID_HEIGHT + 2, n_possible_actions),
                               dtype=np.float16)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def draw_action(self, obstacle_position: Position, player_position: Position) -> int:
        available_actions = [
            Action.STAY.value,
            Action.RIGHT.value,
            Action.LEFT.value
        ]

        if np.random.rand() < self.exploration_rate:
            return np.random.choice(available_actions)
        else:
            return np.argmax(
                self.qtable[obstacle_position.x, obstacle_position.y, player_position.x, player_position.y]
            )

    def update_qtable(self, obstacle_position, player_position, action, reward, new_obstacle_position):
        self.qtable[obstacle_position.x, obstacle_position.y, player_position.x, player_position.y, action] += (
            self.learning_rate * (reward + self.discount_factor * (
            np.max(
                self.qtable[new_obstacle_position.x, new_obstacle_position.y, player_position.x, player_position.y]) -
            self.qtable[
                obstacle_position.x, obstacle_position.y, player_position.x, player_position.y, action]))
        )



class QLearningOOO:

    def __init__(
        self,
        n_possible_actions: int,
        exploration_rate: float = 0.3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9
    ):
        # one couple grid_width, grid_height is for the obstacle and the other for the player
        self.qtable = np.zeros((GRID_WIDTH, GRID_HEIGHT, n_possible_actions),
                               dtype=np.float16)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def draw_action(self, obstacle_position: Position, player_position: Position) -> int:

        if np.random.rand() < self.exploration_rate:
            return np.random.choice(range(len(Action)))
        else:
            return np.argmax(
                self.qtable[obstacle_position.x, obstacle_position.y]
            )

    def update_qtable(self, obstacle_position, player_position, action, reward, new_obstacle_position):
        self.qtable[obstacle_position.y, obstacle_position.x, action] += (
            self.learning_rate * (reward + self.discount_factor * (
            np.max(
                self.qtable[new_obstacle_position.y, new_obstacle_position.x]) -
            self.qtable[
                obstacle_position.y, obstacle_position.x, action]))
        )

    def store_qtable(self, path: str):
        np.save(path, self.qtable)

    def load_qtable(self, path: str):
        self.qtable = np.load(path)
