import argparse
import os
import time

import pygame

from casrl.entity.spaceship.adversial_spaceship import AdversarialSpaceship
from casrl.handler.environment_handler import EnvironmentHandler
from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.reward.reward_adversarial_spaceship import RewardAdversarialSpaceship
from casrl.reward.reward_ufo import RewardUFO
from casrl.handler.statistics_handler import StatisticsHandler
from casrl.utils.const import SCREEN_WIDTH, OBSTACLE_SIZE, AGENT_SIZE, SCREEN_HEIGHT, ROOT_DIR


def start_train(
    store_state_path_dir_name: str | None,
    load_state_path_dir_name: str | None,
    n_ufos: int,
    episodes: int
):
    if store_state_path_dir_name is None:
        store_state_path_dir_name = time.strftime("%Y%m%d-%H%M")
    store_path = os.path.join(ROOT_DIR, f"statics/states/{store_state_path_dir_name}")

    pygame.init()
    pygame.display.set_caption("Collision Avoidance Simulation")
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    reward_player = RewardAdversarialSpaceship(positive_reward=100, negative_reward=-100, no_op_reward=-10)
    reward_ufo = RewardUFO(positive_reward=100, negative_reward=-100, no_op_reward=-1e-10)

    agent = AdversarialSpaceship(size=AGENT_SIZE, reward_function=reward_player)
    ufo_collection = UFOCollection(n_ufos=n_ufos, obstacle_size=OBSTACLE_SIZE, reward_function=reward_ufo)
    environment = EnvironmentHandler(ufo_collection, agent, window)

    if load_state_path_dir_name is not None:
        # load state from a preatrained path
        load_path = os.path.join(ROOT_DIR, f"statics/states/{load_state_path_dir_name}")
        environment.load_rl_state(load_path)

    statistics = StatisticsHandler.instance()

    for episode in range(episodes):
        agent.reset()
        ufo_collection.reset(agent.position)

        statistics.episode = episode
        while True:
            clock.tick(statistics.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # check for key presses
            keys = pygame.key.get_pressed()
            statistics.handle_fps_update(keys)

            closer_ufo_position = agent.get_other_position(ufo_collection)
            agent_outcome, is_terminal_agent = agent.run_iteration(closer_ufo_position)
            ufos_outcome, is_terminal_ufos = ufo_collection.run_iteration(agent.position)
            environment.visualize_environment(load_pics=False)

            statistics.handle_outcome_stats_update(ufos_outcome, agent_outcome)

            if is_terminal_agent or is_terminal_ufos:
                break

    environment.save_rl_state(store_path)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Collision Avoidance System PoC",
        description="This script is used to train a number of qlearning agent to avoid collisions from another agent. "
                    "The agent trying to escape will be UFOs, while the agent trying to collide with them will be a"
                    "spaceship. The training will be done with an adversarial setting.",
    )
    parser.add_argument("-sp", "--save-path-to", type=str,
                        help="Path to the the folder containing the state of the agent. If not given, it will default "
                             "to the current time in YYYYMMDD-HHMM format",
                        dest="state_save_path")

    parser.add_argument("-lp", "--load-pretrained-from", type=str, default=None,
                        help="Path to the the folder containing the pretrained state of the agent. This is useful when"
                             "we want to start train from an already trained episode",
                        dest="state_load_path")

    parser.add_argument("-nu", "--n-ufos", type=int, default=1,
                        help="Number of ufos to spawn",
                        dest="n_ufos")

    parser.add_argument("-e", "--episodes", type=int, default=10000,
                        help="Number of episodes to train for",
                        dest="episodes")

    args = parser.parse_args()

    start_train(
        store_state_path_dir_name=args.state_save_path,
        load_state_path_dir_name=args.state_load_path,
        n_ufos=args.n_ufos,
        episodes=args.episodes
    )
