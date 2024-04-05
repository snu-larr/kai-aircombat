import math
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition
from ..utils.utils import LLA2NEU
import numpy as np

class UnreachHeadingSAM(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    # max_heading_increment: 180,       # degree
    # max_altitude_increment: 7000,     # feet
    # max_velocities_u_increment: 100,  # meter
    # check_interval: 30,               # second

    def __init__(self, config):
        super().__init__(config)
        self.max_heading_increment = 180
        self.max_altitude_increment = 7000
        self.max_velocities_u_increment = 100
        self.check_interval = 30
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10

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

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        ego_obs = np.array(env.agents[agent_id].get_property_values(self.state_var))

        first_sam_id = [k for k in env.sams.keys()][0] 
        sam_obs = np.array(env.sams[first_sam_id].get_property_values(self.sam_state_var))

        ego_neu = LLA2NEU(*ego_obs[:3])
        sam_neu = LLA2NEU(*sam_obs)

        delta_alt = sam_neu[2] -  ego_neu[2]
        
        # vector 내적    
        delta_n, delta_e, delta_u = sam_neu[0] - ego_neu[0], sam_neu[1] - ego_neu[1], sam_neu[2] - ego_neu[2]
        proj_dist = delta_n * ego_obs[6] + delta_e * ego_obs[7] + delta_u * ego_obs[8]

        delta_value = math.sqrt(delta_n ** 2 + delta_e ** 2 + delta_u ** 2)
        vel_value = math.sqrt(ego_obs[6] ** 2 + ego_obs[7] ** 2 + ego_obs[8] ** 2)
        delta_heading = math.acos(proj_dist / max(0.0001, (delta_value * vel_value)))        

        done = False
        success = False
        cur_step = info['current_step']
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)
        # check heading when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > 10:
                done = True
            # # if current target heading is reached, random generate a new target heading
            # else:
            #     delta = self.increment_size[env.heading_turn_counts]
            #     delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
            #     delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
            #     delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
            #     new_heading = env.agents[agent_id].get_property_value(c.target_heading_deg) + delta_heading
            #     new_heading = (new_heading + 360) % 360
            #     new_altitude = env.agents[agent_id].get_property_value(c.target_altitude_ft) + delta_altitude
            #     new_velocities_u = env.agents[agent_id].get_property_value(c.target_velocities_u_mps) + delta_velocities_u
            #     env.agents[agent_id].set_property_value(c.target_heading_deg, new_heading)
            #     env.agents[agent_id].set_property_value(c.target_altitude_ft, new_altitude)
            #     env.agents[agent_id].set_property_value(c.target_velocities_u_mps, new_velocities_u)
            #     env.agents[agent_id].set_property_value(c.heading_check_time, check_time + self.check_interval)
            #     env.heading_turn_counts += 1
            #     self.log(f'current_step:{cur_step} target_heading:{new_heading} '
            #              f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')
        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
            info['heading_turn_counts'] = env.heading_turn_counts
        success = False
        return done, success, info
