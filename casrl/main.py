import time

import pygame

from casrl.entity.agent import Agent
from casrl.entity.environment import Environment
from casrl.entity.obstacles import Obstacles
from casrl.entity.outcome import Outcome
from casrl.entity.reward import Reward
from casrl.entity.statistics import Statistics
from const import SCREEN_WIDTH, EPISODES, OBSTACLE_SIZE, AGENT_SIZE, SCREEN_HEIGHT, GRID_WIDTH, GRID_HEIGHT, ROOT_DIR

LOAD_STATE = False

now = time.strftime("%Y%m%d-%H%M")
store_path = ROOT_DIR + f"/statics/states/{now}"

load_path = ROOT_DIR + f"/statics/states/20240322-1626"

pygame.init()
pygame.display.set_caption("Collision Avoidance Simulation")
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

reward_function = Reward(positive_reward=100, negative_reward=-100, no_op_reward=-10)

agent = Agent(size=AGENT_SIZE)
obstacles = Obstacles(n_obstacles=1, obstacle_size=OBSTACLE_SIZE, reward_function=reward_function)

environment = Environment(obstacles, agent, window)

if LOAD_STATE:
    environment.load_rl_state(load_path)

statistics = Statistics.instance()

for episode in range(EPISODES):
    agent.reset_to_fixed_pos()
    obstacles.reset_to_fixed_pos()

    statistics.episode = episode
    episode_duration = 0
    while True:
        dt = clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # check for key presses
        keys = pygame.key.get_pressed()

        environment.visualize_environment()
        agent.run_iteration(keys)
        iteration_outcome = obstacles.run_iteration(agent.position)

        if Outcome.WIN.value in iteration_outcome:
            statistics.n_win += 1
            statistics.episode_durations_win.append(episode_duration)
            break
        if Outcome.OOO.value in iteration_outcome:
            statistics.n_ooo += 1
            break
        if Outcome.COL.value in iteration_outcome:
            statistics.n_col += 1
            break

        episode_duration += 1

environment.save_rl_state(store_path)

# Close Pygame
pygame.quit()
