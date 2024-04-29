import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import body_ned_to_world_ned, KNOT2METER, DEG2RAD

class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = None

    def reset(self, task, env):
        self.previous_missile_v = None
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is velocity attenuation of the missile

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        
        # TODO : reward 는 rwr 의 missile 을 보고 해야하나? 아니면 전지적 시점의 missile 위치를 보고 계산해야하나?
        if (len(env.agents[agent_id].nearest_munition) > 0):
            missile_sim = env.agents[agent_id].nearest_munition

            # lla, deg, knot/s
            missile_lon, missile_lat, missile_alt, missile_r, missile_p, missile_y, missile_v = missile_sim
            missile_v = missile_v * KNOT2METER

            aircraft_v = env.agents[agent_id].get_velocity()
            if self.previous_missile_v is None:
                self.previous_missile_v = missile_v
            v_decrease = (self.previous_missile_v - missile_v) / 340 * self.reward_scale

            missile_heading_direction = body_ned_to_world_ned(1, 0, 0, missile_r * DEG2RAD, missile_p * DEG2RAD, missile_y * DEG2RAD)
            angle = np.dot(missile_heading_direction, aircraft_v) / (np.linalg.norm(missile_heading_direction) * np.linalg.norm(aircraft_v))
            if angle < 0:
                reward = angle / (max(v_decrease, 0) + 1)
            else:
                reward = angle * max(v_decrease, 0)
        else:
            self.previous_missile_v = None
            reward = 0
        self.reward_trajectory[agent_id].append([reward])
        return reward
