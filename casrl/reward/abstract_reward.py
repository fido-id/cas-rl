
from abc import ABC, abstractmethod

from casrl.entity.abstract_agent import AbstractAgent


class AbstractReward(ABC):

    @abstractmethod
    def __init__(self, positive_reward: float | int, negative_reward: float | int, no_op_reward: float | int) -> None:
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.no_op_reward = no_op_reward

    @abstractmethod
    def compute_reward(
        self,
        prev_self_agent: AbstractAgent,
        current_self_agent: AbstractAgent,
        other_agent: AbstractAgent,
    ) -> float:
        pass
