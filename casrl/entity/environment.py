import numpy as np
import pygame

from casrl.const import WHITE, BLACK, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_WIDTH, GRID_HEIGHT
from casrl.entity.agent.abstract_agent import AbstractAgent
from casrl.entity.obstacles import Obstacles
from casrl.entity.statistics import Statistics
from casrl.enums.action import Action


class Environment:

    def __init__(self, obstacles: Obstacles, agent: AbstractAgent, window: pygame.Surface):
        self.obstacles = obstacles
        self.agent = agent
        self.window = window

        self.font = pygame.font.SysFont("Arial", 20)

    def visualize_environment(self):
        self.window.fill(WHITE)
        background = pygame.image.load('statics/images/background.jpg').convert()
        background = pygame.transform.smoothscale(background, self.window.get_size())
        self.window.blit(background, (0, 0))

        spaceship = pygame.image.load('statics/images/spaceship.png').convert_alpha()
        spaceship = pygame.transform.smoothscale(
            spaceship,
            (
                self.agent.position.size * (SCREEN_WIDTH // GRID_WIDTH),
                self.agent.position.size * (SCREEN_WIDTH // GRID_WIDTH)
            )
        )
        # create a rect with the size of the image.
        rect = spaceship.get_rect(
            topleft=(
                self.agent.position.x * (SCREEN_WIDTH // GRID_WIDTH),
                self.agent.position.y * (SCREEN_HEIGHT // GRID_HEIGHT)
            )
        )
        self.window.blit(spaceship, rect)

        for obstacle in self.obstacles.obstacles:
            ufo = pygame.image.load('statics/images/ufo.png').convert_alpha()
            ufo = pygame.transform.smoothscale(
                ufo,
                (
                    obstacle.position.size * (SCREEN_WIDTH // GRID_WIDTH),
                    obstacle.position.size * (SCREEN_WIDTH // GRID_WIDTH)
                )
            )

            # create a rect with the size of the image.
            rect = ufo.get_rect(
                topleft=(
                    obstacle.position.x * (SCREEN_WIDTH // GRID_WIDTH),
                    obstacle.position.y * (SCREEN_HEIGHT // GRID_HEIGHT)
                )
            )
            self.window.blit(ufo, rect)

        statistics = Statistics.instance()
        self.window.blit(self.font.render(f"Episodes: {statistics.episode}", True, BLACK), (0, 0))
        self.window.blit(self.font.render(f"Number STAY: {statistics.n_of_stay_action}", True, BLACK), (0, 20))
        self.window.blit(self.font.render(f"Number LEFT: {statistics.n_of_left_action}", True, BLACK), (0, 40))
        self.window.blit(self.font.render(f"Number RIGHT: {statistics.n_of_right_action}", True, BLACK), (0, 60))
        self.window.blit(self.font.render(f"Number UP: {statistics.n_of_up_action}", True, BLACK), (0, 80))
        self.window.blit(self.font.render(f"Number DOWN: {statistics.n_of_down_action}", True, BLACK), (0, 100))
        self.window.blit(self.font.render(f"Number WIN: {statistics.n_win}", True, BLACK), (0, 120))
        self.window.blit(self.font.render(f"Number OOO pla: {statistics.n_ooo_player}", True, BLACK), (0, 140))
        self.window.blit(self.font.render(f"Number OOO npc: {statistics.n_ooo_npc}", True, BLACK), (0, 160))
        self.window.blit(self.font.render(f"Number COL: {statistics.n_col}", True, BLACK), (0, 180))
        self.window.blit(self.font.render(f"FPS: {statistics.fps}", True, BLACK), (0, 200))
        self.window.blit(self.font.render(
            f"MA(10) Iterations Before Collision: {round(np.average(statistics.n_iterations_before_collision[:-10]), 2)}",
            True, BLACK), (0, 220))
        pygame.display.flip()

    def save_rl_state(self, path):
        self.obstacles.save_qtables(path)

    def load_rl_state(self, path):
        self.obstacles.load_qtables(path)
