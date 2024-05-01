import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, KNOT2METER, body_ned_to_world_ned, DEG2RAD


class MultipleCombatTask(SingleCombatTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 4

    @property
    def num_ai_agents(self) -> int:
        return 2

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
            # c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            # c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            # c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
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
        self.obs_length = 9 + (self.num_agents - 1) * 6 + 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        # self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

        # gun_trigger, aim9_trigger, aim120_trigger, chaff_flare_trigger, jammer_trigger, radar_trigger, target_idx
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30, 2, 2, 2, 2, 2, 2, 2])
        # self.action_space = spaces.Tuple([
        #     spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2),
        #     spaces.Discrete(2), spaces.Discrete(2),
        #     spaces.Discrete(2), spaces.Discrete(2),
        #     spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)
        # ])
        # self.stick_action_dim = self.action_space[0].shape[0]

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)
        
        # TODO : 현재는 전지적 시점에서 적군기 위치까지 고려함
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset+1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset+2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset+3] = AO
            norm_obs[offset+4] = TA
            norm_obs[offset+5] = R / 10000
            norm_obs[offset+6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        
        # (3) relative missile state
        if (len(env.agents[agent_id].nearest_munition) > 0):
            missile_lon, missile_lat, missile_alt, missile_r, missile_p, missile_y, missile_v = env.agents[agent_id].nearest_munition
            missile_v = missile_v * KNOT2METER

            missile_neu = LLA2NEU(missile_lon, missile_lat, missile_alt, env.center_lon, env.center_lat, env.center_alt)
            
            # TODO : missile VN, VE, VD 갑 수신 가능한지 확인해보기
            missile_heading_direction = body_ned_to_world_ned(1, 0, 0, missile_r * DEG2RAD, missile_p * DEG2RAD, missile_y * DEG2RAD)
            missile_vn = missile_heading_direction[0] * missile_v
            missile_ve = missile_heading_direction[1] * missile_v
            missile_vd = missile_heading_direction[2] * missile_v
            
            missile_feature = [missile_neu[0], missile_neu[1], -missile_neu[2], missile_vn, missile_ve, missile_vd]

            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[offset + 1] = (missile_v - ego_state[9]) / 340
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
        return norm_obs
    

    def quick_get_obs(self, env, agent_id):
        norm_obs = np.zeros(9)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs[0] = ego_state[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_state[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_state[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_state[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_state[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_state[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_state[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_state[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_state[12] / 340            # 8. ego vc   (unit: mh)(unit: 5G)

        # norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        self._shoot_action[agent_id] = action[4] # action[5]

        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info
        
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        
        super().reset(env)

    def step(self, env):
        SingleCombatTask.step(self, env)
