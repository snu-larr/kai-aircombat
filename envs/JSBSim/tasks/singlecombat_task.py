import numpy as np
from gymnasium import spaces
from collections import deque
from ..core.catalog import Catalog as c
from .task_base import BaseTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward
from ..utils.utils import LLA2NEU, get_AO_TA_R, KNOT2METER, DEG2RAD, body_ned_to_world_ned
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn

class SingleCombatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            # EventDrivenReward(self.config),
            # ShootPenaltyReward(self.config)
        ]
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            SafeReturn(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def num_ai_agents(self) -> int:
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
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        # self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])
        
        # aileron, elevator, rudder, throttle, gun attack, missile AIM9/120 attack, chaff/Flare attack, Jammer On, Radar On, target
        # self.action_space = spaces.MultiDiscrete([41, 41, 41, 30, 2, 2, 2, 2, 2, 2, 2, 2])
        self.action_space = spaces.Tuple([
            spaces.MultiDiscrete([41, 41, 41, 30]), 
            spaces.Discrete(2), spaces.Discrete(2),
            spaces.Discrete(2), spaces.Discrete(2),
            spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)
        ])
        self.stick_action_dim = self.action_space[0].shape[0]


    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(9 + 6 + 6)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag

        # (3) relative missile state
        if (len(env.agents[agent_id].nearest_munition) > 0):
            missile_lon, missile_lat, missile_alt, missile_r, missile_p, missile_y, missile_v = env.agents[agent_id].nearest_munition
            missile_v = missile_v * KNOT2METER

            missile_neu = LLA2NEU(missile_lon, missile_lat, missile_alt, env.center_lon, env.center_lat, env.center_alt)
            
            # TODO : missile VN, VE, VD
            missile_heading_direction = body_ned_to_world_ned(1, 0, 0, missile_r * DEG2RAD, missile_p * DEG2RAD, missile_y * DEG2RAD)
            missile_vn = missile_heading_direction[0] * missile_v
            missile_ve = missile_heading_direction[1] * missile_v
            missile_vd = missile_heading_direction[2] * missile_v
            
            missile_feature = [missile_neu[0], missile_neu[1], -missile_neu[2], missile_vn, missile_ve, missile_vd]

            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (missile_v - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag

        return norm_obs
    
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[self.stick_action_dim]

        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20  - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        
        self._agent_die_flag = {}

        super().reset(env)
    
    def step(self, env):
        pass

    def get_reward(self, env, agent_id, info=...):
        if self._agent_die_flag.get(agent_id, False):
            return 0.0, info
        else:
            self._agent_die_flag[agent_id] = not env.agents[agent_id].is_alive
            return super().get_reward(env, agent_id, info=info)