from typing import Any

import numpy as np
import pytest

from casrl.entity.position import Position
from casrl.enums.action import Action
from mocks.mock_agent import MockAgent


@pytest.mark.parametrize("drawn_int, expected_action", [
    [0, Action.STAY],
    [1, Action.LEFT],
    [2, Action.RIGHT],
    [3, Action.UP],
    [4, Action.DOWN],
])
def test_action_selection(qlearning_instance, mocker, drawn_int, expected_action):
    mocker.patch('numpy.random.rand', return_value=0.2)
    mocker.patch('numpy.argmax', return_value=drawn_int)
    self_agent = MockAgent(position=Position(5, 5), size=1)
    other_agent = MockAgent(position=Position(7, 7), size=1)
    action = qlearning_instance.draw_action(self_agent, other_agent)
    assert action == expected_action


def test_update_qtable(qlearning_instance):
    prev_self_agent = MockAgent(position=Position(4, 4), size=1)
    other_agent = MockAgent(position=Position(7, 7), size=1)
    action = Action.LEFT
    reward = 1.0
    current_self_agent = MockAgent(position=Position(3, 4), size=1)
    is_terminal_state = False
    qlearning_instance.update_qtable(prev_self_agent, other_agent, action, reward, current_self_agent, is_terminal_state)
    assert np.allclose(qlearning_instance.qtable[45, Action.LEFT.value], 0.1)


def test_store_and_load_qtable(qlearning_instance, tmp_path):
    path = tmp_path / "qtable.npy"
    qlearning_instance.store_qtable(path)
    assert path.exists()
    qlearning_instance.load_qtable(path)
    assert np.array_equal(qlearning_instance.qtable, np.load(path))
