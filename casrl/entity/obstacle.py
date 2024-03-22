import copy

import numpy as np

from casrl.action import Action
from casrl.const import GRID_HEIGHT, GRID_WIDTH, MOVEMENT_OFFSET
from casrl.entity.outcome import Outcome
from casrl.entity.position import Position
from casrl.entity.reward import Reward
from casrl.entity.statistics import Statistics
from casrl.qlearning import QLearning


class Obstacle:
    def __init__(self, size: int, reward_function: Reward):
        self.position = None
        self.size = size
        self.reward_function = reward_function
        self.q = QLearning(n_possible_actions=len(Action))

    def run_iteration(self, player_position: Position) -> bool:
        prev_position = copy.copy(self.position)
        action = self.q.draw_action(self.position, player_position)
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

        reward, outcome = self.reward_function.compute_reward_and_outcome(self.position, player_position)

        is_terminal_state = self.reward_function.is_terminal(reward)
        self.q.update_qtable(
            prev_position, player_position, action, reward, self.position, is_terminal_state
        )

        return outcome.value

    def reset(self, agent_position: Position) -> None:

        # do not reset obstacle inside the agent
        potential_y_positions = np.arange(1, GRID_HEIGHT - self.size)
        agent_y_coordinates = np.arange(agent_position.y, agent_position.y + agent_position.size)
        mask_y = np.isin(potential_y_positions, agent_y_coordinates)
        potential_y_positions = potential_y_positions[~mask_y]

        potential_x_positions = np.arange(1, GRID_WIDTH - self.size)
        agent_x_coordinates = np.arange(agent_position.x, agent_position.x + agent_position.size)
        mask_x = np.isin(potential_x_positions, agent_x_coordinates)
        potential_x_positions = potential_x_positions[~mask_x]

        self.position = Position(
            np.random.choice(potential_x_positions),
            np.random.choice(potential_y_positions),
            self.size
        )

    def save_qtable(self, path: str) -> None:
        self.q.store_qtable(path)

    def load_qtable(self, path: str) -> None:
        self.q.load_qtable(path)


