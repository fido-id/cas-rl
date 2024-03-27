from typing import Tuple

from casrl.utils.const import GRID_WIDTH, GRID_HEIGHT
from casrl.enums.outcome import Outcome
from casrl.entity.position import Position
from casrl.reward.abstract_reward import AbstractReward


class RewardUFO(AbstractReward):

    def __init__(self, positive_reward: int, negative_reward: int, no_op_reward: float) -> None:
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.no_op_reward = no_op_reward

    def compute_reward(self, prev_self_position: Position, self_position: Position, other_position: Position) -> float:
        if self_position.is_out_of_bounds():
            # check if the agent went out of bound
            return self.no_op_reward
        elif self_position.overlaps_with(other_position):
            # check for collision
            return self.negative_reward

        new_distance = self_position.distance_from(other_position)
        old_distance = prev_self_position.distance_from(other_position)
        if new_distance <= old_distance:
            return self.negative_reward
        return self.positive_reward

