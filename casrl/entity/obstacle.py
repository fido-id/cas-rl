import copy

import numpy as np

from casrl.action import Action
from casrl.const import GRID_HEIGHT, GRID_WIDTH, MOVEMENT_OFFSET
from casrl.entity.outcome import Outcome
from casrl.entity.position import Position
from casrl.entity.statistics import Statistics
from casrl.qlearning import QLearning, QLearningOOO


class Obstacle:
    def __init__(self, x: int, y: int, size: int):
        self.position = Position(x, y, size)
        self.q = QLearningOOO(n_possible_actions=len(Action))

    def run_iteration(self, player_position: Position) -> bool:
        prev_position = copy.copy(self.position)
        action = self.q.draw_action(self.position, player_position)
        statistics = Statistics.instance()
        match action:
            case Action.RIGHT.value:
                self.position.update(MOVEMENT_OFFSET, 0)
                statistics.n_of_right_action += 1
            case Action.LEFT.value:
                self.position.update(-MOVEMENT_OFFSET, 0)
                statistics.n_of_left_action += 1
            case Action.STAY.value:
                statistics.n_of_stay_action += 1
            case Action.DOWN.value:
                self.position.update(0, MOVEMENT_OFFSET)
                statistics.n_of_down_action += 1
            case Action.UP.value:
                self.position.update(0, -MOVEMENT_OFFSET)
                statistics.n_of_up_action += 1
                pass

        reward = 0
        if (self.position.x <= 0 or self.position.x >= GRID_WIDTH - self.position.size or
            self.position.y <= 0 or self.position.y >= GRID_HEIGHT - self.position.size):
            # check if the obstacle went out of bound
            reward = -100
        elif self.position.overlaps(player_position):
            # check for collision
            reward = 100

        self.q.update_qtable(
            prev_position, player_position, action, reward, self.position
        )

        if reward == 100:
            return Outcome.WIN.value
        elif reward < -1:
            return Outcome.OOO.value

        return Outcome.NOOP.value

    def reset(self, x: int, y: int) -> None:
        self.position = Position(x, y, self.position.size)

    def save_qtable(self, path: str) -> None:
        self.q.store_qtable(path)

    def load_qtable(self, path: str) -> None:
        self.q.load_qtable(path)


