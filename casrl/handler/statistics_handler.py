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
    episode_durations_win = [0]
    n_ooo_player = 0
    n_ooo_npc = 0
    n_col = 0
    n_iterations_before_collision = [0]
    fps = 120

    def __init__(self):
        raise RuntimeError("Call instance() method instead")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def handle_fps_update(self, keys):
        if keys[pygame.K_q]:
            self.fps = max(self.fps - 10, 1)
        if keys[pygame.K_e]:
            self.fps = self.fps + 10

    def handle_outcome_stats_update(self, ufos_outcome: list[Outcome], agent_outcome: Outcome):
        if Outcome.COL in ufos_outcome or agent_outcome == Outcome.COL:
            self.n_col += 1

        if Outcome.OOO in ufos_outcome:
            self.n_ooo_npc += 1

        if agent_outcome == Outcome.OOO:
            self.n_ooo_player += 1










