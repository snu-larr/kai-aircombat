import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingSAMReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading, UnreachHeadingSAM
from ..utils.utils import LLA2NEU
import math

class SAMTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingSAMReward(self.config),
            AltitudeReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeadingSAM(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
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
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

        # # aileron, elevator, rudder, throttle, gun attack, missile AIM9/120 attack, chaff/Flare attack, Jammer On, Radar On, target
        # # self.action_space = spaces.MultiDiscrete([41, 41, 41, 30, 2, 2, 2, 2, 2, 2, 2, 2])
        # self.action_space = spaces.Tuple([
        #     spaces.MultiDiscrete([41, 41, 41, 30]), 
        #     spaces.Discrete(2), spaces.Discrete(2),
        #     spaces.Discrete(2), spaces.Discrete(2),
        #     spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)
        # ])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        ego_obs = np.array(env.agents[agent_id].get_property_values(self.state_var))

        first_sam_id = [k for k in env.sams.keys()][0] 
        sam_obs = np.array(env.sams[first_sam_id].get_property_values(self.sam_state_var))

        ego_neu = LLA2NEU(*ego_obs[:3])
        sam_neu = LLA2NEU(*sam_obs)

        delta_alt = ego_neu[2] - sam_neu[2]
        
        # vector 내적    
        delta_n, delta_e, delta_u = sam_neu[0] - ego_neu[0], sam_neu[1] - ego_neu[1], sam_neu[2] - ego_neu[2]
        proj_dist = delta_n * ego_obs[6] + delta_e * ego_obs[7]

        delta_value = math.sqrt(delta_n ** 2 + delta_e ** 2)
        vel_value = math.sqrt(ego_obs[6] ** 2 + ego_obs[7] ** 2)
        delta_heading = math.acos(proj_dist / max(0.0001, (delta_value * vel_value)))

        norm_obs = np.zeros(12)
        norm_obs[0] = delta_alt / 1000          # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = delta_heading             # 1. ego delta heading  (unit rad)
        norm_obs[2] = ego_obs[9] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = ego_obs[2] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(ego_obs[3])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(ego_obs[3])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(ego_obs[4])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(ego_obs[4])        # 7. ego_pitch_cos
        norm_obs[8] = ego_obs[6] / 340          # 8. ego_v_north    (unit: mh)
        norm_obs[9] = ego_obs[7] / 340          # 9. ego_v_east     (unit: mh)
        norm_obs[10] = ego_obs[8] / 340         # 10. ego_v_down    (unit: mh)
        norm_obs[11] = ego_obs[12] / 340         # 11. ego_vc        (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act
