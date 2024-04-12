import os

from casrl.entity.abstract_agent import AbstractAgent
from casrl.entity.ufo.ufo_entity import UFO
from casrl.enums.outcome import Outcome
from casrl.reward.reward_ufo import RewardUFO


class UFOCollection:
    def __init__(
        self, n_ufos: int, obstacle_size: int, reward_function: RewardUFO | None = None, exploration_rate: float = 0.3
    ) -> None:

        self.ufos: list[UFO] = [
            UFO(obstacle_size, reward_function, exploration_rate=exploration_rate)
            for _ in range(n_ufos)
        ]
        self.obstacle_size = obstacle_size

    def run_iteration(self, other_agent: AbstractAgent) -> tuple[list[Outcome], bool]:
        outcomes = []
        is_terminals = []
        for ufo in self.ufos:
            outcome, is_terminal = ufo.run_iteration(other_agent)
            is_terminals.append(is_terminal)
            outcomes.append(outcome)

        return outcomes, any(is_terminals)

    def reset(self, other_agent: AbstractAgent) -> None:
        for ufo in self.ufos:
            ufo.reset(other_agent)

    def save_qtables(self, root_path: str) -> None:
        os.makedirs(root_path, exist_ok=True)
        for i, ufo in enumerate(self.ufos):
            obstacle_state_path = f"{root_path}/{i}.npy"
            ufo.save_qtable(obstacle_state_path)
            print(f"Saved ufo {i} state to {obstacle_state_path}")

    def load_qtables(self, root_path: str) -> None:
        if not os.path.exists(root_path):
            print(f"Cannot load RL state from {root_path}")
            return

        for i, ufo in enumerate(self.ufos):
            ufo.load_qtable(f"{root_path}/{i}.npy")

        print(f"Loaded RL states from {root_path}")
