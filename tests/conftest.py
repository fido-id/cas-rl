import pytest

from casrl.utils.qlearning import QLearning


@pytest.fixture
def qlearning_instance():
    return QLearning(
        state_size=(360,),
        n_possible_actions=4,
        exploration_rate=0.1,
        learning_rate=0.1,
        discount_factor=0.9
    )
