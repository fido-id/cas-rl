import numpy as np

from casrl.utils.const import MOVEMENT_OFFSET, GRID_WIDTH, GRID_HEIGHT
from casrl.enums.action import Action
from casrl.entity.position import Position


class QLearning:

    def __init__(
        self,
        state_size: tuple,
        n_possible_actions: int,
        exploration_rate: float = 0.3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9
    ):
        self.qtable = np.zeros((*state_size, n_possible_actions), dtype=np.float32)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def draw_action(self, self_position: Position, other_position: Position, is_deployed: bool = False) -> int:
        angle = self_position.angle_from(other_position)
        if angle == 360:
            angle = 0

        potential_actions = []
        # do not allow actions that bring the agent out of bounds
        if self_position.x - MOVEMENT_OFFSET >= 0:
            potential_actions.append(Action.LEFT.value)
        if self_position.x + MOVEMENT_OFFSET <= GRID_WIDTH - self_position.size:
            potential_actions.append(Action.RIGHT.value)
        if self_position.y - MOVEMENT_OFFSET >= 0:
            potential_actions.append(Action.UP.value)
        if self_position.y + MOVEMENT_OFFSET <= GRID_HEIGHT - self_position.size:
            potential_actions.append(Action.DOWN.value)

        if np.random.rand() < self.exploration_rate and not is_deployed:
            return np.random.choice(potential_actions)
        else:
            return np.argmax(
                self.qtable[angle]
            )

    def update_qtable(
        self, prev_self_position, other_position, action, reward, self_position, is_terminal_state
    ):
        angle = prev_self_position.angle_from(other_position)
        new_angle = self_position.angle_from(other_position)
        if angle == 360:
            angle = 0
        if new_angle == 360:
            new_angle = 0

        max_expected_reward = self.discount_factor * np.max(self.qtable[new_angle])
        if is_terminal_state:
            max_expected_reward = 0
        self.qtable[angle, action] += self.learning_rate * (reward + max_expected_reward - self.qtable[angle, action])

    def store_qtable(self, path: str):
        np.save(path, self.qtable)

    def load_qtable(self, path: str):
        self.qtable = np.load(path)
