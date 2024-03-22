import pygame

from casrl.const import WHITE, RED, BLACK, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_WIDTH, GRID_HEIGHT
from casrl.entity.agent import Agent
from casrl.entity.obstacles import Obstacles
from casrl.entity.statistics import Statistics


class Environment:

    def __init__(self, obstacles: Obstacles, agent: Agent, window: pygame.Surface):
        self.obstacles = obstacles
        self.agent = agent
        self.window = window

        self.font = pygame.font.SysFont("Arial", 20)

    def visualize_environment(self):
        self.window.fill(WHITE)
        pygame.draw.rect(self.window, RED, (
            self.agent.position.x * (SCREEN_WIDTH // GRID_WIDTH),
            self.agent.position.y * (SCREEN_HEIGHT // GRID_HEIGHT),
            self.agent.position.size * (SCREEN_WIDTH // GRID_WIDTH),
            self.agent.position.size * (SCREEN_WIDTH // GRID_WIDTH)
        ))
        pygame.draw.rect(self.window, RED, (
            self.agent.position.x * (SCREEN_WIDTH // GRID_WIDTH),
            self.agent.position.y * (SCREEN_HEIGHT // GRID_HEIGHT),
            self.agent.position.size * (SCREEN_WIDTH // GRID_WIDTH),
            self.agent.position.size * (SCREEN_WIDTH // GRID_WIDTH)
        ))
        for obstacle in self.obstacles.obstacles:
            pygame.draw.rect(self.window, BLACK, (
                obstacle.position.x * (SCREEN_WIDTH // GRID_WIDTH),
                obstacle.position.y * (SCREEN_HEIGHT // GRID_HEIGHT),
                obstacle.position.size * (SCREEN_WIDTH // GRID_WIDTH),
                obstacle.position.size * (SCREEN_WIDTH // GRID_WIDTH)
            ))

        statistics = Statistics.instance()
        self.window.blit(self.font.render(f"Episodes: {statistics.episode}", True, BLACK), (0, 0))
        self.window.blit(self.font.render(f"Number STAY: {statistics.n_of_stay_action}", True, BLACK), (0, 20))
        self.window.blit(self.font.render(f"Number LEFT: {statistics.n_of_left_action}", True, BLACK), (0, 40))
        self.window.blit(self.font.render(f"Number RIGHT: {statistics.n_of_right_action}", True, BLACK), (0, 60))
        self.window.blit(self.font.render(f"Number UP: {statistics.n_of_up_action}", True, BLACK), (0, 80))
        self.window.blit(self.font.render(f"Number DOWN: {statistics.n_of_down_action}", True, BLACK), (0, 100))
        self.window.blit(self.font.render(f"Number WIN: {statistics.n_win}", True, BLACK), (0, 120))
        self.window.blit(self.font.render(f"Number OOO: {statistics.n_ooo}", True, BLACK), (0, 140))
        self.window.blit(self.font.render(f"Episode duration: {max(statistics.episode_durations)}", True, BLACK), (0, 160))
        pygame.display.flip()

    def save_rl_state(self, path):
        self.obstacles.save_qtables(path)

    def load_rl_state(self, path):
        self.obstacles.load_qtables(path)
