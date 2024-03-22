import time

import pygame

from casrl.entity.agent import Agent
from casrl.entity.environment import Environment
from casrl.entity.obstacles import Obstacles
from casrl.entity.outcome import Outcome
from casrl.entity.statistics import Statistics
from const import SCREEN_WIDTH, EPISODES, OBSTACLE_SIZE, AGENT_SIZE, SCREEN_HEIGHT, GRID_WIDTH, GRID_HEIGHT, ROOT_DIR


now = time.strftime("%Y%m%d-%H%M")
store_path = ROOT_DIR + f"/statics/states/{now}"

load_path = ROOT_DIR + f"/statics/states/20240322-0841"

pygame.init()
pygame.display.set_caption("Collision Avoidance Simulation")
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

obstacles = Obstacles(
    n_obstacles=1, obstacle_size=OBSTACLE_SIZE
)
agent = Agent(initial_x=int(GRID_WIDTH / 2), initial_y=15, size=AGENT_SIZE)

environment = Environment(obstacles, agent, window)

environment.load_rl_state(load_path)

statistics = Statistics.instance()

for episode in range(EPISODES):
    agent.reset()
    obstacles.reset()

    statistics.episode = episode
    episode_duration = 0
    while True:
        dt = clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # check for key presses
        keys = pygame.key.get_pressed()

        environment.visualize_environment()
        agent.run_episode(keys)
        iteration_outcome = obstacles.run_iteration(agent.position)

        if Outcome.WIN.value in iteration_outcome:
            statistics.n_win += 1
            break
        if Outcome.OOO.value in iteration_outcome:
            statistics.n_ooo += 1
            break

        episode_duration += 1

    statistics.episode_durations.append(episode_duration)

environment.save_rl_state(store_path)

# Close Pygame
pygame.quit()
