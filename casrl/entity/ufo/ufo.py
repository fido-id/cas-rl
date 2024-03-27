import numpy as np

from casrl.entity.abstract_trainable_agent import AbstractTrainableAgent
from casrl.enums.outcome import Outcome
from casrl.utils.const import GRID_HEIGHT, GRID_WIDTH, CAS_THRESHOLD
from casrl.entity.position import Position
from casrl.reward.reward_ufo import RewardUFO


class UFO(AbstractTrainableAgent):
    def __init__(
        self,
        size: int,
        reward_function: RewardUFO,
        exploration_rate: float,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
    ):
        super().__init__(
            size, reward_function, exploration_rate, learning_rate, discount_factor
        )

    def get_other_position(self, player_position: Position) -> Position:
        return player_position

    def run_iteration(self, other_position: Position) -> tuple[Outcome, bool]:
        if self._position.distance_from(other_position) > CAS_THRESHOLD:
            # don't move if the obstacle is far away
            return Outcome.NOOP.value, False
        return super().run_iteration(other_position)

    def reset(self, agent_position: Position) -> None:
        # do not reset obstacle inside the agent to minimize bias during learning
        potential_y_positions = np.arange(1, GRID_HEIGHT - self.size)
        agent_y_coordinates = np.arange(agent_position.y, agent_position.y + agent_position.size)
        mask_y = np.isin(potential_y_positions, agent_y_coordinates)
        potential_y_positions = potential_y_positions[~mask_y]

        potential_x_positions = np.arange(1, GRID_WIDTH - self.size)
        agent_x_coordinates = np.arange(agent_position.x, agent_position.x + agent_position.size)
        mask_x = np.isin(potential_x_positions, agent_x_coordinates)
        potential_x_positions = potential_x_positions[~mask_x]

        self._position = Position(
            np.random.choice(potential_x_positions),
            np.random.choice(potential_y_positions),
            self.size
        )
