import enum


class Outcome(enum.Enum):
    OOO = 1
    NOOP = 2
    COL = 3

    @staticmethod
    def is_terminal(outcome: "Outcome") -> bool:
        if outcome == Outcome.OOO or outcome == Outcome.COL:
            return True
        return False
