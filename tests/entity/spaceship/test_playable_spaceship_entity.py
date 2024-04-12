import numpy as np
import pygame
import pytest

from casrl.entity.position import Position
from casrl.entity.spaceship.playable_spaceship_entity import PlayableSpaceship


def test_playable_spaceship_run_iteration_no_movement(mocker):
    keys = mocker.MagicMock(spec=dict)
    keys.__getitem__.side_effect = lambda key: False
    playable_spaceship = PlayableSpaceship(size=1)
    playable_spaceship.position = Position(0, 0)

    playable_spaceship.run_iteration(keys)

    assert playable_spaceship.position.x == 0
    assert playable_spaceship.position.y == 0


@pytest.mark.parametrize("pressed_key, expected_delta_x, expected_delta_y", [
    [pygame.K_w, 0, -1],
    [pygame.K_s, 0, 1],
    [pygame.K_a, -1, 0],
    [pygame.K_d, 1, 0],
])
def test_playable_spaceship_run_iteration_movement(mocker, pressed_key, expected_delta_x, expected_delta_y):
    keys = mocker.MagicMock(spec=dict)
    keys.__getitem__.side_effect = lambda key: True if key in [pressed_key] else False
    playable_spaceship = PlayableSpaceship(size=1)

    initial_x = 5
    initial_y = 5
    playable_spaceship.position = Position(initial_x, initial_y)
    playable_spaceship.run_iteration(keys)

    assert playable_spaceship.position.x == initial_x + expected_delta_x
    assert playable_spaceship.position.y == initial_y + expected_delta_y


def test_reset(mocker):
    mocker.patch.object(np.random, 'randint', side_effect=[5, 6])
    playable_spaceship = PlayableSpaceship(size=1)
    # Reset method is already called in the initializer, no need to call it explicitly
    assert playable_spaceship.position.x == 5
    assert playable_spaceship.position.y == 6