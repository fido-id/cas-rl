import math
from typing import Tuple

from casrl.const import GRID_WIDTH, GRID_HEIGHT
from casrl.entity.outcome import Outcome
from casrl.entity.position import Position


class Reward:

    def __init__(self, positive_reward: int, negative_reward: int, no_op_reward: int) -> None:
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.no_op_reward = no_op_reward

    def is_terminal(self, reward) -> bool:
        if reward == self.positive_reward or reward == self.negative_reward:
            return True
        return False

    def compute_reward_and_outcome(self, obstacle_position: Position, player_position: Position) -> Tuple[int, Outcome]:
        if (obstacle_position.x <= 0 or obstacle_position.x >= GRID_WIDTH - obstacle_position.size or
                obstacle_position.y <= 0 or obstacle_position.y >= GRID_HEIGHT - obstacle_position.size):
            # check if the obstacle went out of bound
            return self.negative_reward, Outcome.OOO
        elif obstacle_position.overlaps(player_position):
            # check for collision
            return self.positive_reward, Outcome.WIN
        distance = obstacle_position.distance(player_position)
        diagonal = math.ceil((GRID_WIDTH**2 + GRID_HEIGHT**2)**(1/2))
        discounted_reward = distance / diagonal
        return self.no_op_reward * discounted_reward, Outcome.NOOP

