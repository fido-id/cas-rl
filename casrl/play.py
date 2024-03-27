import argparse
import os

import pygame

from casrl.entity.spaceship.playable_spaceship import PlayableSpaceship
from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.enums.outcome import Outcome
from casrl.handler.environment_handler import EnvironmentHandler
from casrl.handler.statistics_handler import StatisticsHandler
from casrl.utils.const import SCREEN_WIDTH, OBSTACLE_SIZE, AGENT_SIZE, SCREEN_HEIGHT, ROOT_DIR


def start_play(load_state_path_dir_name: str):
    pygame.init()
    pygame.display.set_caption("Collision Avoidance Simulation")
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    load_path = ROOT_DIR + f"/statics/states/{load_state_path_dir_name}"

    n_ufos = len(os.listdir(load_path))

    agent = PlayableSpaceship(size=AGENT_SIZE)
    # exploration rate here is set to zero since we don't need randomisation anymore
    ufo_collection = UFOCollection(n_ufos=n_ufos, obstacle_size=OBSTACLE_SIZE, exploration_rate=0)
    environment = EnvironmentHandler(ufo_collection, agent, window)
    environment.load_rl_state(load_path)

    statistics_handler = StatisticsHandler.instance()

    while True:
        agent.reset()
        ufo_collection.reset(agent.position)

        while True:
            clock.tick(statistics_handler.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # check for key presses
            keys = pygame.key.get_pressed()

            statistics_handler.handle_fps_update(keys)

            agent.run_iteration(keys)
            ufos_outcome, is_terminal_ufos = ufo_collection.run_iteration(agent.position)
            environment.visualize_environment()

            statistics_handler.handle_outcome_stats_update(ufos_outcome, Outcome.NOOP)

            if is_terminal_ufos:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Collision Avoidance System PoC",
        description="This script is used to play with a trained agent. The latter must avoid collision from a player "
                    "that can freely move inside a predefined grid. When the UFO goes out of bound the game restarts"
    )

    parser.add_argument("-lp", "--load-pretrained-from", type=str, required=True,
                        help="Path to the the folder containing the pretrained state of the agent.",
                        dest="state_load_path")

    args = parser.parse_args()

    start_play(
        load_state_path_dir_name=args.state_load_path,
    )
