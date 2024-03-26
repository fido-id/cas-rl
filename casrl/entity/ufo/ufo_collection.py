import os

from casrl.entity.position import Position
from casrl.entity.ufo.ufo import UFO
from casrl.reward.reward_ufo import RewardUFO


class UFOCollection:
    def __init__(self, n_ufos: int, obstacle_size: int, reward_function: RewardUFO):

        self.ufos = [
            UFO(obstacle_size, reward_function)
            for _ in range(n_ufos)
        ]
        self.obstacle_size = obstacle_size

    def run_iteration(self, player_position: Position, is_deployed: bool):
        outcomes = []
        for obstacle in self.ufos:
            outcomes.append(obstacle.run_iteration(player_position, is_deployed))

        return outcomes

    def reset(self, agent_position: Position):
        for obstacle in self.ufos:
            obstacle.reset(agent_position)

    def reset_to_fixed_pos(self):
        for obstacle in self.ufos:
            obstacle.reset_to_fixed_pos()

    def save_qtables(self, root_path):
        os.makedirs(root_path, exist_ok=True)
        for i, obstacle in enumerate(self.ufos):
            obstacle_state_path = f"{root_path}/{i}.npy"
            obstacle.save_qtable(obstacle_state_path)
            print(f"Saved obstacle {i} state to {obstacle_state_path}")

    def load_qtables(self, root_path):
        if not os.path.exists(root_path):
            print(f"Cannot load RL state from {root_path}")
            return

        for i, obstacle in enumerate(self.ufos):
            obstacle.load_qtable(f"{root_path}/{i}.npy")

        print(f"Loaded RL states from {root_path}")
