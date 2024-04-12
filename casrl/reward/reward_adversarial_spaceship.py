from casrl.entity.abstract_agent import AbstractAgent
from casrl.reward.abstract_reward import AbstractReward


class RewardAdversarialSpaceship(AbstractReward):

    def __init__(self, positive_reward: int, negative_reward: int, no_op_reward: int) -> None:
        super().__init__(positive_reward, negative_reward, no_op_reward)

    def compute_reward(
        self,
        prev_self_agent: AbstractAgent,
        current_self_agent: AbstractAgent,
        other_agent: AbstractAgent,
    ) -> float:
        if current_self_agent.is_out_of_bounds():
            # check if the agent went out of bound
            return self.negative_reward
        elif current_self_agent.overlaps_with(other_agent):
            # check for collision
            return self.positive_reward

        new_distance = current_self_agent.distance_from(other_agent)
        old_distance = prev_self_agent.distance_from(other_agent)
        if new_distance < old_distance:
            return self.positive_reward
        return self.no_op_reward

