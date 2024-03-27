import os

from casrl.entity.position import Position
from casrl.entity.ufo.ufo import UFO
from casrl.enums.outcome import Outcome
from casrl.reward.reward_ufo import RewardUFO


class UFOCollection:
    def __init__(
        self, n_ufos: int, obstacle_size: int, reward_function: RewardUFO | None = None, exploration_rate: float = 0.3
    ):

        self.ufos = [
            UFO(obstacle_size, reward_function, exploration_rate=exploration_rate)
            for _ in range(n_ufos)
        ]
        self.obstacle_size = obstacle_size

    def run_iteration(self, player_position: Position) -> tuple[list[Outcome], bool]:
        outcomes = []
        is_terminals = []
        for ufo in self.ufos:
            outcome, is_terminal = ufo.run_iteration(player_position)
            is_terminals.append(is_terminal)
            outcomes.append(outcome)

        return outcomes, any(is_terminals)

    def reset(self, agent_position: Position):
        for ufo in self.ufos:
            ufo.reset(agent_position)

    def save_qtables(self, root_path):
        os.makedirs(root_path, exist_ok=True)
        for i, ufo in enumerate(self.ufos):
            obstacle_state_path = f"{root_path}/{i}.npy"
            ufo.save_qtable(obstacle_state_path)
            print(f"Saved ufo {i} state to {obstacle_state_path}")

    def load_qtables(self, root_path):
        if not os.path.exists(root_path):
            print(f"Cannot load RL state from {root_path}")
            return

        for i, ufo in enumerate(self.ufos):
            ufo.load_qtable(f"{root_path}/{i}.npy")

        print(f"Loaded RL states from {root_path}")
