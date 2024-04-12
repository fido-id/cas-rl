import numpy as np

from casrl.entity.position import Position
from casrl.entity.spaceship.adversial_spaceship_entity import AdversarialSpaceship
from casrl.entity.ufo.ufo_collection import UFOCollection
from casrl.entity.ufo.ufo_entity import UFO


def test_adversarial_spaceship_get_closest_ufo(mocker):
    ufo1 = UFO(size=1, reward_function=None, exploration_rate=0.1)
    ufo1.position = Position(0, 0)
    ufo2 = UFO(size=1, reward_function=None, exploration_rate=0.1)
    ufo2.position = Position(10, 10)
    ufo_collection = UFOCollection(n_ufos=2, obstacle_size=1)
    ufo_collection.ufos = [ufo1, ufo2]
    adversarial_spaceship_instance = AdversarialSpaceship(size=1, reward_function=mocker.Mock(), exploration_rate=0)
    adversarial_spaceship_instance.position = Position(1,1)
    closest_ufo = adversarial_spaceship_instance.get_closest_ufo(ufo_collection)
    assert closest_ufo == ufo1


def test_reset(mocker):
    mocker.patch.object(np.random, 'randint', side_effect=[5, 6])
    adversarial_spaceship = AdversarialSpaceship(size=1, reward_function=mocker.Mock(), exploration_rate=0)

    adversarial_spaceship.reset()

    assert adversarial_spaceship.position.x == 5
    assert adversarial_spaceship.position.y == 6
