import os

import numpy as np

from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.entity.position import Position
from casrl.enums.outcome import Outcome
from mocks.mock_agent import MockAgent


def test_run_iteration(mocker):
    other_agent = MockAgent(position=Position(10, 10), size=1)
    ufo_collection = UFOCollection(n_ufos=3, obstacle_size=1, reward_function=None, exploration_rate=0.1)

    mocker.patch.object(ufo_collection.ufos[0], 'run_iteration', return_value=(Outcome.COL, False))
    mocker.patch.object(ufo_collection.ufos[1], 'run_iteration', return_value=(Outcome.NOOP, False))
    mocker.patch.object(ufo_collection.ufos[2], 'run_iteration', return_value=(Outcome.OOO, True))

    outcomes, is_terminal = ufo_collection.run_iteration(other_agent)

    assert outcomes == [Outcome.COL, Outcome.NOOP, Outcome.OOO]
    assert is_terminal


def test_reset(mocker):
    other_agent = MockAgent(position=Position(5, 5), size=1)
    ufo_collection = UFOCollection(n_ufos=3, obstacle_size=1, reward_function=None, exploration_rate=0.1)

    # Mock reset method of UFO instances
    mocker.patch.object(ufo_collection.ufos[0], 'reset')
    mocker.patch.object(ufo_collection.ufos[1], 'reset')
    mocker.patch.object(ufo_collection.ufos[2], 'reset')

    ufo_collection.reset(other_agent)

    ufo_collection.ufos[0].reset.assert_called_once_with(other_agent)
    ufo_collection.ufos[1].reset.assert_called_once_with(other_agent)
    ufo_collection.ufos[2].reset.assert_called_once_with(other_agent)


def test_save_load_qtables(mocker, tmp_path):
    root_path = tmp_path / "qtables"
    root_path.mkdir()
    ufo_collection = UFOCollection(n_ufos=3, obstacle_size=1, reward_function=None, exploration_rate=0.1)

    # Mock save_qtable and load_qtable methods of UFO instances
    ufo_collection.save_qtables(str(root_path))
    assert len(os.listdir(root_path)) == 3

    ufo_collection.load_qtables(str(root_path))
    assert np.array_equal(ufo_collection.ufos[0].q.qtable, np.load(root_path / "0.npy"))
    assert np.array_equal(ufo_collection.ufos[1].q.qtable, np.load(root_path / "1.npy"))
    assert np.array_equal(ufo_collection.ufos[2].q.qtable, np.load(root_path / "2.npy"))

