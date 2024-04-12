import copy
from abc import ABC, abstractmethod

from casrl.entity.abstract_agent import AbstractAgent
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
    ) -> None:
        super().__init__(size)
        self.reward_function = reward_function
        self.q = QLearning(
            state_size=(360,),
            n_possible_actions=len(Action),
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def run_iteration(self, other_agent: AbstractAgent) -> tuple[Outcome, bool]:

        statistics = StatisticsHandler.instance()
        prev_agent_state = copy.deepcopy(self)
        action = self.q.draw_action(self, other_agent)
        match action:
            case Action.RIGHT:
                self.position.update(MOVEMENT_OFFSET, 0)
                statistics.n_of_right_action += 1
            case Action.LEFT:
                self.position.update(-MOVEMENT_OFFSET, 0)
                statistics.n_of_left_action += 1
            case Action.STAY:
                statistics.n_of_stay_action += 1
            case Action.DOWN:
                self.position.update(0, MOVEMENT_OFFSET)
                statistics.n_of_down_action += 1
            case Action.UP:
                self.position.update(0, -MOVEMENT_OFFSET)
                statistics.n_of_up_action += 1
                pass

        outcome = self.get_iteration_outcome(other_agent)
        is_terminal_state = Outcome.is_terminal(outcome)

        # if no reward function is given, it means we are in a "deploying" state where we just need to
        # follow the policy without modifying it
        if self.reward_function is not None:
            reward = self.reward_function.compute_reward(prev_agent_state, self, other_agent)
            self.q.update_qtable(
                prev_agent_state, other_agent, action, reward, self, is_terminal_state
            )

        return outcome, is_terminal_state

    def get_iteration_outcome(self, other_agent: AbstractAgent) -> Outcome:
        if self.is_out_of_bounds():
            # check if the current agent went out of bound
            return Outcome.OOO
        elif self.overlaps_with(other_agent):
            # check for collision
            return Outcome.COL
        return Outcome.NOOP

    def save_qtable(self, path: str) -> None:
        self.q.store_qtable(path)

    def load_qtable(self, path: str) -> None:
        self.q.load_qtable(path)

