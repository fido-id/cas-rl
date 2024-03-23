import copy
import math

import numpy as np

from casrl.enums.action import Action
from casrl.const import GRID_HEIGHT, GRID_WIDTH, MOVEMENT_OFFSET, CAS_THRESHOLD
from casrl.enums.outcome import Outcome
from casrl.entity.position import Position
from casrl.reward.abstract_reward import AbstractReward
from casrl.entity.statistics import Statistics
from casrl.qlearning import QLearning
from casrl.reward.reward_npc import RewardNPC


class Obstacle:
    def __init__(self, size: int, reward_function: RewardNPC):
        self.position = None
        self.size = size
        self.reward_function = reward_function
        self.q = QLearning(
            state_size=(360,), n_possible_actions=len(Action)
        )
        
        self.n_iterations_before_collision = 0

    def run_iteration(self, player_position: Position, is_deployed: bool) -> int:
        if self.position.distance_from(player_position) > CAS_THRESHOLD:
            return Outcome.NOOP.value

        statistics = Statistics.instance()
        prev_position = copy.copy(self.position)
        action = self.q.draw_action(self.position, player_position, is_deployed=is_deployed)
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

        reward, outcome = self.reward_function.compute_reward_and_outcome(prev_position, self.position, player_position)
        
        if outcome.value != Outcome.COL.value:
            self.n_iterations_before_collision += 1
        else:
            statistics.n_iterations_before_collision.append(self.n_iterations_before_collision)
            self.n_iterations_before_collision = 0
        
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

    def reset_to_fixed_pos(self) -> None:
        self.position = Position(
            np.random.randint(4, GRID_WIDTH - self.size - 4),
            GRID_HEIGHT - 12,
            self.size
        )

    def save_qtable(self, path: str) -> None:
        self.q.store_qtable(path)

    def load_qtable(self, path: str) -> None:
        self.q.load_qtable(path)


