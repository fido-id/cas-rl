import numpy as np

from casrl.entity.abstract_agent import AbstractAgent
from casrl.entity.abstract_trainable_agent import AbstractTrainableAgent
from casrl.entity.position import Position
from casrl.enums.outcome import Outcome
from casrl.reward.reward_ufo import RewardUFO
from casrl.utils.const import CAS_THRESHOLD, GRID_HEIGHT, GRID_WIDTH


class UFO(AbstractTrainableAgent):
    def __init__(
        self,
        size: int,
        reward_function: RewardUFO | None,
        exploration_rate: float,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
    ) -> None:
        super().__init__(
            size, reward_function, exploration_rate, learning_rate, discount_factor
        )

    def run_iteration(self, other_agent: AbstractAgent) -> tuple[Outcome, bool]:
        if self.distance_from(other_agent) > CAS_THRESHOLD:
            # don't move if the obstacle is far away
            return Outcome.NOOP, False
        return super().run_iteration(other_agent)

    def reset(self, other_agent: AbstractAgent) -> None:
        # do not reset obstacle inside the agent to minimize bias during learning
        potential_y_positions = np.arange(1, GRID_HEIGHT - self.size)
        agent_y_coordinates = np.arange(other_agent.position.y - self.size, other_agent.position.y + other_agent.size)
        mask_y = np.isin(potential_y_positions, agent_y_coordinates)
        potential_y_positions = potential_y_positions[~mask_y]

        potential_x_positions = np.arange(1, GRID_WIDTH - self.size)
        agent_x_coordinates = np.arange(other_agent.position.x - self.size, other_agent.position.x + other_agent.size)
        mask_x = np.isin(potential_x_positions, agent_x_coordinates)
        potential_x_positions = potential_x_positions[~mask_x]

        x_index = np.random.randint(len(potential_x_positions))
        y_index = np.random.randint(len(potential_y_positions))
        self.position = Position(
            potential_x_positions[x_index].item(),
            potential_y_positions[y_index].item(),
        )
