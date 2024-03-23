from typing import Tuple

from casrl.enums.outcome import Outcome
from casrl.entity.position import Position
from abc import ABC, abstractmethod


class AbstractReward(ABC):

    @abstractmethod
    def __init__(self, positive_reward: int, negative_reward: int, no_op_reward: int) -> None:
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.no_op_reward = no_op_reward

    def is_terminal(self, reward) -> bool:
        if reward == self.positive_reward or reward == self.negative_reward:
            return True
        return False

    @abstractmethod
    def compute_reward_and_outcome(self, prev_self_position: Position, self_position: Position, other_position: Position) -> Tuple[int, Outcome]:
        pass