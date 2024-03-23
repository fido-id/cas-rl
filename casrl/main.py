import time

import pygame

from casrl.entity.agent.adversial_agent import AdversarialAgent
from casrl.entity.agent.playable_agent import PlayableAgent
from casrl.entity.environment import Environment
from casrl.entity.obstacles import Obstacles
from casrl.enums.outcome import Outcome
from casrl.reward.reward_player import RewardPlayer
from casrl.reward.reward_npc import RewardNPC
from casrl.entity.statistics import Statistics
from const import SCREEN_WIDTH, EPISODES, OBSTACLE_SIZE, AGENT_SIZE, SCREEN_HEIGHT, ROOT_DIR

LOAD_STATE = True
PLAY = True

now = time.strftime("%Y%m%d-%H%M")
store_path = ROOT_DIR + f"/statics/states/{now}"

load_path = ROOT_DIR + f"/statics/states/20240323-1740"

pygame.init()
pygame.display.set_caption("Collision Avoidance Simulation")
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

reward_player = RewardPlayer(positive_reward=100, negative_reward=-100, no_op_reward=-10)
reward_npc = RewardNPC(positive_reward=100, negative_reward=-100, no_op_reward=-0.00000000001)

if PLAY:
    agent = PlayableAgent(size=AGENT_SIZE)
else:
    agent = AdversarialAgent(size=AGENT_SIZE, reward_function=reward_player)
obstacles = Obstacles(n_obstacles=2, obstacle_size=OBSTACLE_SIZE, reward_function=reward_npc)

environment = Environment(obstacles, agent, window)

if LOAD_STATE:
    environment.load_rl_state(load_path)

statistics = Statistics.instance()

for episode in range(EPISODES):
    agent.reset()
    obstacles.reset(agent.position)

    statistics.episode = episode
    episode_duration = 0
    while True:
        dt = clock.tick(statistics.fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # check for key presses
        keys = pygame.key.get_pressed()

        if keys[pygame.K_q]:
            statistics.fps = max(statistics.fps - 5, 1)
        if keys[pygame.K_e]:
            statistics.fps = statistics.fps + 5

        if PLAY:
            agent_outcome = agent.run_iteration(keys)
        else:
            agent_outcome = agent.run_iteration(obstacles)

        iteration_outcome = obstacles.run_iteration(agent.position, PLAY)
        environment.visualize_environment()

        if Outcome.WIN.value in iteration_outcome or agent_outcome == Outcome.WIN.value:
            statistics.n_win += 1
            statistics.episode_durations_win.append(episode_duration)
            break
        if Outcome.OOO.value in iteration_outcome:
            statistics.n_ooo_npc += 1
            break
        if agent_outcome == Outcome.OOO.value:
            statistics.n_ooo_player += 1
            break
        if Outcome.COL.value in iteration_outcome or agent_outcome == Outcome.COL.value:
            statistics.n_col += 1
            break

        episode_duration += 1

environment.save_rl_state(store_path)

# Close Pygame
pygame.quit()
