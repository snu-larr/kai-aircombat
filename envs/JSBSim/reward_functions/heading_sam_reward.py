import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
import numpy as np
from ..utils.utils import LLA2NEU

DEG2RAD = 3.14159265/180
RAD2DEG = 180/3.14159265

class HeadingSAMReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]

        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
        ]
        self.sam_state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
        ]

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_obs = np.array(env.agents[agent_id].get_property_values(self.state_var))

        first_sam_id = [k for k in env.sams.keys()][0] 
        sam_obs = np.array(env.sams[first_sam_id].get_property_values(self.sam_state_var))

        ego_neu = LLA2NEU(*ego_obs[:3])
        sam_neu = LLA2NEU(*sam_obs)

        delta_alt = ego_obs[2] - sam_obs[2]

        # vector 내적    
        delta_n, delta_e, delta_u = sam_neu[0] - ego_neu[0], sam_neu[1] - ego_neu[1], sam_neu[2] - ego_neu[2]
        proj_dist = delta_n * ego_obs[6] + delta_e * ego_obs[7]

        delta_value = math.sqrt(delta_n ** 2 + delta_e ** 2)
        vel_value = math.sqrt(ego_obs[6] ** 2 + ego_obs[7] ** 2)
        delta_heading = math.acos(proj_dist / max(0.0001, (delta_value * vel_value)))        


        heading_error_scale = 5.0  # degrees
        heading_r = math.exp(-((delta_heading * RAD2DEG / heading_error_scale) ** 2))

        alt_error_scale = 15.24  # m
        alt_r = math.exp(-((delta_alt / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((env.agents[agent_id].get_property_value(c.attitude_roll_rad) / roll_error_scale) ** 2))

        # speed_error_scale = 24  # mps (~10%)
        # speed_r = math.exp(-((ego_obs[9] / speed_error_scale) ** 2))
        speed_r = 1
        
        # print("REWARD : ", heading_r, alt_r, roll_r, speed_r)

        reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r))
