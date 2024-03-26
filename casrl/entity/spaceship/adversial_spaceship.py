import copy

import numpy as np

from casrl.entity.spaceship.abstract_spaceship import AbstractSpaceship
from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.enums.action import Action
from casrl.utils.const import GRID_HEIGHT, GRID_WIDTH, MOVEMENT_OFFSET
from casrl.entity.position import Position
from casrl.reward.abstract_reward import AbstractReward
from casrl.handler.statistics import Statistics
from casrl.utils.qlearning import QLearning


class AdversarialSpaceship(AbstractSpaceship):
    def __init__(self, size: int, reward_function: AbstractReward):
        self._position = None
        self.size = size
        self.reward_function = reward_function
        self.q = QLearning(state_size=(360,), n_possible_actions=len(Action))

    def run_iteration(self, ufo_collection: UFOCollection) -> int:
        closer_ufo_pos = ufo_collection.ufos[0].position
        distance_from_closer_ufo = float("inf")
        for ufo in ufo_collection.ufos:
            distance_from_ufo = ufo.position.distance_from(self.position)
            if distance_from_ufo < distance_from_closer_ufo:
                closer_ufo_pos = ufo.position
                distance_from_closer_ufo = distance_from_ufo

        prev_position = copy.copy(self.position)
        action = self.q.draw_action(self.position, closer_ufo_pos)
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

        reward, outcome = self.reward_function.compute_reward_and_outcome(prev_position, self.position, closer_ufo_pos)

        is_terminal_state = self.reward_function.is_terminal(reward)
        self.q.update_qtable(
            prev_position, closer_ufo_pos, action, reward, self.position, is_terminal_state
        )

        return outcome.value

    def reset(self) -> None:
        self._position = Position(
            np.random.randint(1, GRID_WIDTH - self.size),
            np.random.randint(1, GRID_HEIGHT - self.size),
            self.size
        )

    def reset_to_fixed_pos(self) -> None:
        self._position = Position(
            np.random.randint(4, GRID_WIDTH - self.size - 4),
            GRID_HEIGHT - 15,
            self.size
        )

    def save_qtable(self, path: str) -> None:
        self.q.store_qtable(path)

    def load_qtable(self, path: str) -> None:
        self.q.load_qtable(path)

    @property
    def position(self) -> Position:
        return self._position


