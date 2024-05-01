from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
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
        reward = 0
        if env.agents[agent_id].is_shotdown:
            reward -= 200
        elif env.agents[agent_id].is_crash:
            reward -= 200

        # Launch 한 missile 이 격추를 성공했을 때
        for mu_id, target_id_dmg_dict in env.mu_id_target_id_dmg.items():
            # agent 가 laucnh 한 missile 인 경우
            # 피격 판정은 한 tick 에서만 일어남
            if (env.mu_id_upid[mu_id] == agent_id):
                reward += 200

        return self._process(reward, agent_id)
