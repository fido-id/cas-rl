import pygame

from casrl.enums.outcome import Outcome


class StatisticsHandler:

    _instance = None
    episode = 0
    n_of_left_action = 0
    n_of_right_action = 0
    n_of_stay_action = 0
    n_of_up_action = 0
    n_of_down_action = 0
    n_ooo_spaceship = 0
    n_ooo_ufo = 0
    n_col = 0
    fps = 120

    def __init__(self) -> None:
        raise RuntimeError("Call instance() method instead")

    @classmethod
    def instance(cls) -> "StatisticsHandler":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def handle_fps_update(self, keys: pygame.key.ScancodeWrapper) -> None:
        if keys[pygame.K_q]:
            self.fps = max(self.fps - 10, 1)
        if keys[pygame.K_e]:
            self.fps = self.fps + 10

    def handle_outcome_stats_update(self, ufos_outcome: list[Outcome], spaceship_outcome: Outcome) -> None:
        if Outcome.COL in ufos_outcome or spaceship_outcome == Outcome.COL:
            self.n_col += 1

        if Outcome.OOO in ufos_outcome:
            self.n_ooo_ufo += 1

        if spaceship_outcome == Outcome.OOO:
            self.n_ooo_spaceship += 1










