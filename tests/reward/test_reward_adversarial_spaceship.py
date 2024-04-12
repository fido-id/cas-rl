from casrl.entity.position import Position
from casrl.reward.reward_adversarial_spaceship import RewardAdversarialSpaceship
from mocks.mock_agent import MockAgent


def test_out_of_bounds():
    reward_function = RewardAdversarialSpaceship(positive_reward=1, negative_reward=-1, no_op_reward=0)
    prev_self_agent = MockAgent(position=Position(1, 1), size=1)
    current_self_agent = MockAgent(position=Position(0, 1), size=1)
    other_agent = MockAgent(position=Position(5, 5), size=1)
    reward = reward_function.compute_reward(prev_self_agent, current_self_agent, other_agent)
    assert reward == -1


def test_collision():
    reward_function = RewardAdversarialSpaceship(positive_reward=1, negative_reward=-1, no_op_reward=0)
    prev_self_agent = MockAgent(position=Position(1, 1), size=1)
    current_self_agent = MockAgent(position=Position(1, 2), size=1)
    other_agent = MockAgent(position=Position(1, 2), size=1)
    reward = reward_function.compute_reward(prev_self_agent, current_self_agent, other_agent)
    assert reward == 1


def test_no_change_in_distance():
    reward_function = RewardAdversarialSpaceship(positive_reward=1, negative_reward=-1, no_op_reward=0)
    prev_self_agent = MockAgent(position=Position(2, 2), size=1)
    current_self_agent = MockAgent(position=Position(1, 2), size=1)
    other_agent = MockAgent(position=Position(5, 5), size=1)
    reward = reward_function.compute_reward(prev_self_agent, current_self_agent, other_agent)
    assert reward == 0


def test_closer_to_other_agent():
    reward_function = RewardAdversarialSpaceship(positive_reward=1, negative_reward=-1, no_op_reward=0)
    prev_self_agent = MockAgent(position=Position(2, 2), size=1)
    current_self_agent = MockAgent(position=Position(3, 2), size=1)
    other_agent = MockAgent(position=Position(5, 5), size=1)
    reward = reward_function.compute_reward(prev_self_agent, current_self_agent, other_agent)
    assert reward == 1

