from typing import Any

from casrl.entity.abstract_agent import AbstractAgent
from casrl.enums.outcome import Outcome


class MockAgent(AbstractAgent):
    def run_iteration(self, *args: Any, **kwargs) -> tuple[Outcome, bool] | None:
        pass

    def reset(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __init__(self, position, size):
        self._position = position
        self.size = size
