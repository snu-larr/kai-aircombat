import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.JSBSim.core.catalog import Catalog as c
import numpy as np
import time

def normalize_action(action):
    """Convert discrete action index into continuous value.
    """
    norm_act = np.zeros(4)
    norm_act[0] = action[0] * 2. / 40 - 1.
    norm_act[1] = action[1] * 2. / 40 - 1.
    norm_act[2] = action[2] * 2. / 40 - 1.
    norm_act[3] = action[3] * 0.5 / 29 + 0.4
    return norm_act

action_var = [
    c.fcs_aileron_cmd_norm,
    c.fcs_elevator_cmd_norm,
    c.fcs_rudder_cmd_norm,
    c.fcs_throttle_cmd_norm,
]

agent = AircraftSimulator(
    num_missiles = 0,
    init_state = {
      "ic_long_gc_deg": 120.0,
      "ic_lat_geod_deg": 60.0,
      "ic_h_sl_ft": 20000,
      "ic_psi_true_deg": 0.0,
      "ic_u_fps": 800.0,
    }
)

# @profile
def test():
    iteration = 0
    while True:
        iteration += 1
        action = np.random.randint(10, 30, size = 4)

        a_action = normalize_action(action)
        agent.set_property_values(action_var, a_action)
        print(a_action)
        start_time = time.time()
        agent.run()
        end_time = time.time()

        print(agent._geodetic)

        if (end_time - start_time != 0):
            print("iter : ", iteration, " / JSBSim interaction hz : ", 1 / (end_time - start_time))
        else:
            print("iter : ", iteration, " / Run time Zero!", start_time, end_time)
    
test()