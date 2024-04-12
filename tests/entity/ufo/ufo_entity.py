import numpy as np

from casrl.entity.position import Position
from casrl.entity.ufo.ufo_entity import UFO
from casrl.reward.reward_ufo import RewardUFO
from casrl.utils.const import GRID_WIDTH, GRID_HEIGHT
from mocks.mock_agent import MockAgent


def test_reset(mocker):
    ufo = UFO(size=3, reward_function=None, exploration_rate=0.1)
    other_agent = MockAgent(position=Position(10, 10), size=4)

    # we will test all possible positions. However, we need to take into account that all the possible position is not
    # the entire grid since each time we remove from all the possible positions the ufo (two times, one because of the
    # out of bounds and the other because we want to spawn the ufo in a couple (x,y (representigng the top-left corner
    # corner of the rectangle)) such that the entire rectangle doesn't collide the agent.
    for i in np.arange(0, GRID_WIDTH - ufo.size * 2 - other_agent.size - 1):
        for j in np.arange(0, GRID_HEIGHT - ufo.size * 2 - other_agent.size - 1):
            mocker.patch('numpy.random.randint', side_effect=[i, j])
            ufo.reset(other_agent)
            assert 1 <= ufo.position.x < GRID_WIDTH - ufo.size
            assert 1 <= ufo.position.y < GRID_HEIGHT - ufo.size
            assert not ufo.overlaps_with(other_agent), f"UFO overlaps with Agent, AgentPos=({other_agent.position.x}, {other_agent.position.y}), UFOPos=({ufo.position.x}, {ufo.position.y})"


