import os
from typing import Tuple

import pygame

from casrl.entity.abstract_agent import AbstractAgent
from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.handler.statistics_handler import StatisticsHandler
from casrl.utils.const import BLACK, GRID_HEIGHT, GRID_WIDTH, RED, ROOT_DIR, SCREEN_HEIGHT, SCREEN_WIDTH, WHITE


class EnvironmentHandler:

    def __init__(self, ufo_collection: UFOCollection, spaceship: AbstractAgent, window: pygame.Surface) -> None:
        self.ufo_collection = ufo_collection
        self.spaceship = spaceship
        self.window = window

        self.background_graphics = pygame.image.load(
            os.path.join(ROOT_DIR, "statics/images/background.jpg")
        ).convert()

        self.spaceship_graphics = pygame.image.load(
            os.path.join(ROOT_DIR, "statics/images/spaceship.png")
        ).convert_alpha()

        self.ufo_graphics = pygame.image.load(
            os.path.join(ROOT_DIR, "statics/images/ufo.png")
        ).convert_alpha()

        self.font = pygame.font.SysFont("Arial", 20)

    def visualize_environment(self, load_pics: bool = True) -> None:
        self.window.fill(WHITE)

        if load_pics:
            background = pygame.transform.smoothscale(self.background_graphics, self.window.get_size())
            self.window.blit(background, (0, 0))

            spaceship, rect = self.get_surface_and_rect(self.spaceship_graphics, self.spaceship)
            self.window.blit(spaceship, rect)

            for _ufo in self.ufo_collection.ufos:
                ufo, rect = self.get_surface_and_rect(self.ufo_graphics, _ufo)
                self.window.blit(ufo, rect)
        else:
            pygame.draw.rect(self.window, RED, (
                self.spaceship.position.x * (SCREEN_WIDTH // GRID_WIDTH),
                self.spaceship.position.y * (SCREEN_HEIGHT // GRID_HEIGHT),
                self.spaceship.size * (SCREEN_WIDTH // GRID_WIDTH),
                self.spaceship.size * (SCREEN_WIDTH // GRID_WIDTH)
            ))
            for _ufo in self.ufo_collection.ufos:
                pygame.draw.rect(self.window, BLACK, (
                    _ufo.position.x * (SCREEN_WIDTH // GRID_WIDTH),
                    _ufo.position.y * (SCREEN_HEIGHT // GRID_HEIGHT),
                    _ufo.size * (SCREEN_WIDTH // GRID_WIDTH),
                    _ufo.size * (SCREEN_WIDTH // GRID_WIDTH)
                ))

        statistics = StatisticsHandler.instance()
        self.window.blit(self.font.render(f"Episodes: {statistics.episode}", True, BLACK), (0, 0))
        self.window.blit(self.font.render(f"Number STAY: {statistics.n_of_stay_action}", True, BLACK), (0, 20))
        self.window.blit(self.font.render(f"Number LEFT: {statistics.n_of_left_action}", True, BLACK), (0, 40))
        self.window.blit(self.font.render(f"Number RIGHT: {statistics.n_of_right_action}", True, BLACK), (0, 60))
        self.window.blit(self.font.render(f"Number UP: {statistics.n_of_up_action}", True, BLACK), (0, 80))
        self.window.blit(self.font.render(f"Number DOWN: {statistics.n_of_down_action}", True, BLACK), (0, 100))
        self.window.blit(self.font.render(f"Number OOO spaceship: {statistics.n_ooo_spaceship}", True, BLACK), (0, 120))
        self.window.blit(self.font.render(f"Number OOO ufo: {statistics.n_ooo_ufo}", True, BLACK), (0, 140))
        self.window.blit(self.font.render(f"Number COL: {statistics.n_col}", True, BLACK), (0, 160))
        self.window.blit(self.font.render(f"FPS: {statistics.fps}", True, BLACK), (0, 180))
        pygame.display.flip()

    @staticmethod
    def get_surface_and_rect(graphics: pygame.Surface, agent: AbstractAgent) -> Tuple[pygame.Surface, pygame.Rect]:
        scaled_graphics = pygame.transform.smoothscale(
            graphics,
            (
                agent.size * (SCREEN_WIDTH // GRID_WIDTH),
                agent.size * (SCREEN_WIDTH // GRID_WIDTH)
            )
        )
        # create a rect with the size of the image.
        rect = scaled_graphics.get_rect(
            topleft=(
                agent.position.x * (SCREEN_WIDTH // GRID_WIDTH),
                agent.position.y * (SCREEN_HEIGHT // GRID_HEIGHT)
            )
        )
        return scaled_graphics, rect

    def save_rl_state(self, path: str) -> None:
        self.ufo_collection.save_qtables(path)

    def load_rl_state(self, path: str) -> None:
        self.ufo_collection.load_qtables(path)
