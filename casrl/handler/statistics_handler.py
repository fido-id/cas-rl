
class StatisticsHandler:

    _instance = None
    episode = 0
    n_of_left_action = 0
    n_of_right_action = 0
    n_of_stay_action = 0
    n_of_up_action = 0
    n_of_down_action = 0
    episode_durations_win = [0]
    n_win = 0
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






