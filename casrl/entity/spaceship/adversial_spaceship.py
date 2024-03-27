import numpy as np

from casrl.entity.abstract_trainable_agent import AbstractTrainableAgent
from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.reward.reward_adversarial_spaceship import RewardAdversarialSpaceship
from casrl.utils.const import GRID_HEIGHT, GRID_WIDTH
from casrl.entity.position import Position


class AdversarialSpaceship(AbstractTrainableAgent):

    def __init__(
        self,
        size: int,
        reward_function: RewardAdversarialSpaceship,
        exploration_rate: float = 0.3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
    ):
        super().__init__(
            size, reward_function, exploration_rate, learning_rate, discount_factor
        )

    def get_other_position(self, ufo_collection: UFOCollection) -> Position:
        closer_ufo_pos = ufo_collection.ufos[0].position
        distance_from_closer_ufo = float("inf")
        for ufo in ufo_collection.ufos:
            distance_from_ufo = ufo.position.distance_from(self.position)
            if distance_from_ufo < distance_from_closer_ufo:
                closer_ufo_pos = ufo.position
                distance_from_closer_ufo = distance_from_ufo

        return closer_ufo_pos

    def reset(self) -> None:
        self._position = Position(
            np.random.randint(1, GRID_WIDTH - self.size),
            np.random.randint(1, GRID_HEIGHT - self.size),
            self.size
        )



