from .reward_function_base import BaseRewardFunction


class SAM_Destroy_reward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        first_sam_id = [id for id in env.sams.keys()][0]

        # TODO : 발사할때마다 minus?


        reward = 0
        if env.sams[first_sam_id].is_shotdown:
            reward += 200
        elif env.sams[first_sam_id].is_crash:
            reward += 200

        return self._process(reward, agent_id)
