import copy
from abc import ABC, abstractmethod

from casrl.entity.abstract_agent import AbstractAgent
from casrl.entity.position import Position
from casrl.enums.action import Action
from casrl.enums.outcome import Outcome
from casrl.handler.statistics_handler import StatisticsHandler
from casrl.reward.abstract_reward import AbstractReward
from casrl.utils.const import MOVEMENT_OFFSET
from casrl.utils.qlearning import QLearning


class AbstractTrainableAgent(AbstractAgent, ABC):

    @abstractmethod
    def __init__(
        self,
        size: int,
        reward_function: AbstractReward | None,
        exploration_rate: float,
        learning_rate: float,
        discount_factor: float,
    ):
        super().__init__(size)
        self.reward_function = reward_function
        self.q = QLearning(
            state_size=(360,),
            n_possible_actions=len(Action),
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def get_other_position(self, **kwargs) -> Position:
        pass

    def run_iteration(self, other_position: Position) -> tuple[Outcome, bool]:

        statistics = StatisticsHandler.instance()
        prev_position = copy.copy(self.position)
        action = self.q.draw_action(self.position, other_position)
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

        outcome = self.get_iteration_outcome(self.position, other_position)
        is_terminal_state = Outcome.is_terminal(outcome)

        if self.reward_function is None:
            # if no reward function are given, it means we are in a "deploying" state where we just need to
            # follow the policy without modifying it
            return outcome, is_terminal_state

        reward = self.reward_function.compute_reward(prev_position, self.position, other_position)
        self.q.update_qtable(
            prev_position, other_position, action, reward, self.position, is_terminal_state
        )

        return outcome, is_terminal_state

    @staticmethod
    def get_iteration_outcome(self_position: Position, other_position: Position) -> Outcome:
        if self_position.is_out_of_bounds():
            # check if the agent went out of bound
            return Outcome.OOO
        elif self_position.overlaps_with(other_position):
            # check for collision
            return Outcome.COL
        return Outcome.NOOP

    def save_qtable(self, path: str) -> None:
        self.q.store_qtable(path)

    def load_qtable(self, path: str) -> None:
        self.q.load_qtable(path)

