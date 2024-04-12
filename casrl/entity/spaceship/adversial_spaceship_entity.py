import numpy as np

from casrl.entity.abstract_trainable_agent import AbstractTrainableAgent
from casrl.entity.position import Position
from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.entity.ufo.ufo_entity import UFO
from casrl.reward.reward_adversarial_spaceship import RewardAdversarialSpaceship
from casrl.utils.const import GRID_HEIGHT, GRID_WIDTH


class AdversarialSpaceship(AbstractTrainableAgent):

    def __init__(
        self,
        size: int,
        reward_function: RewardAdversarialSpaceship,
        exploration_rate: float = 0.3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
    ) -> None:
        super().__init__(
            size, reward_function, exploration_rate, learning_rate, discount_factor
        )

    def get_closest_ufo(self, ufo_collection: UFOCollection) -> UFO:
        closest_ufo = ufo_collection.ufos[0]
        distance_from_closest_ufo = float("inf")
        for ufo in ufo_collection.ufos:
            distance_from_ufo = ufo.distance_from(self)
            if distance_from_ufo < distance_from_closest_ufo:
                closest_ufo = ufo
                distance_from_closest_ufo = distance_from_ufo

        return closest_ufo

    def reset(self) -> None:
        self.position = Position(
            np.random.randint(1, GRID_WIDTH - self.size),
            np.random.randint(1, GRID_HEIGHT - self.size),
        )



