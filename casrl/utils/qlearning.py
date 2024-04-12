import numpy as np

from casrl.entity.abstract_agent import AbstractAgent
from casrl.enums.action import Action
from casrl.utils.const import GRID_HEIGHT, GRID_WIDTH, MOVEMENT_OFFSET


class QLearning:

    def __init__(
        self,
        state_size: tuple,
        n_possible_actions: int,
        exploration_rate: float,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        self.qtable = np.zeros((*state_size, n_possible_actions), dtype=np.float32)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def draw_action(self, self_agent: AbstractAgent, other_agent: AbstractAgent) -> Action:
        # angle 0 and 360 must coincide, so we use % operator
        angle = self_agent.angle_from(other_agent) % 360

        potential_actions = []
        # do not allow actions that bring the agent out of bounds
        if self_agent.position.x - MOVEMENT_OFFSET >= 0:
            potential_actions.append(Action.LEFT.value)
        if self_agent.position.x + MOVEMENT_OFFSET <= GRID_WIDTH - self_agent.size:
            potential_actions.append(Action.RIGHT.value)
        if self_agent.position.y - MOVEMENT_OFFSET >= 0:
            potential_actions.append(Action.UP.value)
        if self_agent.position.y + MOVEMENT_OFFSET <= GRID_HEIGHT - self_agent.size:
            potential_actions.append(Action.DOWN.value)

        if np.random.rand() < self.exploration_rate:
            return Action(np.random.choice(potential_actions))
        else:
            return Action(np.argmax(
                self.qtable[angle]
            ))

    def update_qtable(
        self,
        prev_self_agent: AbstractAgent,
        other_agent: AbstractAgent,
        action: Action,
        reward: float,
        current_self_agent: AbstractAgent,
        is_terminal_state: bool
    ) -> None:
        # angle 0 and 360 must coincide, so we use % operator
        prev_angle = prev_self_agent.angle_from(other_agent) % 360
        current_angle = current_self_agent.angle_from(other_agent) % 360

        max_expected_reward = self.discount_factor * np.max(self.qtable[current_angle])
        if is_terminal_state:
            max_expected_reward = 0
        self.qtable[prev_angle, action.value] += (self.learning_rate *
                                                  (reward + max_expected_reward - self.qtable[prev_angle, action.value])
                                                  )

    def store_qtable(self, path: str) -> None:
        np.save(path, self.qtable)

    def load_qtable(self, path: str) -> None:
        self.qtable = np.load(path)
