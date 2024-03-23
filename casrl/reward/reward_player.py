import math
from typing import Tuple

from casrl.const import GRID_WIDTH, GRID_HEIGHT
from casrl.enums.outcome import Outcome
from casrl.entity.position import Position
from casrl.reward.abstract_reward import AbstractReward


class RewardPlayer(AbstractReward):

    def __init__(self, positive_reward: int, negative_reward: int, no_op_reward: int) -> None:
        super().__init__(positive_reward, negative_reward, no_op_reward)

    def compute_reward_and_outcome(self, prev_self_position: Position, self_position: Position, other_position: Position) -> Tuple[int, Outcome]:
        if (self_position.x <= 0 or self_position.x >= GRID_WIDTH - self_position.size or
                self_position.y <= 0 or self_position.y >= GRID_HEIGHT - self_position.size):
            # check if the agent went out of bound
            return self.negative_reward, Outcome.OOO
        elif self_position.overlaps_with(other_position):
            # check for collision
            return self.positive_reward, Outcome.COL

        new_distance = self_position.distance_from(other_position)
        old_distance = prev_self_position.distance_from(other_position)
        if new_distance < old_distance:
            return self.positive_reward, Outcome.NOOP
        return self.negative_reward, Outcome.NOOP
        # distance = self_position.distance_from(other_position)
        # diagonal = math.ceil((GRID_WIDTH**2 + GRID_HEIGHT**2)**(1/2))
        # discounted_reward = (diagonal - distance) / diagonal
        # return self.no_op_reward * discounted_reward, Outcome.NOOP

