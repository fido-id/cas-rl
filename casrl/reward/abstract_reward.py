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

    @abstractmethod
    def compute_reward(self, prev_self_position: Position, self_position: Position, other_position: Position) -> Tuple[int, Outcome]:
        pass
