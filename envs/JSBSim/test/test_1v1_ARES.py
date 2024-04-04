from abc import ABC
import sys
import os
import pandas as pd
import time
import datetime
import math
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir, LLA2NEU, NEU2LLA, body_ned_to_world_ned, hit_rate, damage_rate, get_AO_TA_R, world_ned_to_body_ned, cal_azi_ele_from_euler

# keyboard interrupt
from algorithms.ppo.ppo_actor import PPOActor

FEET2METER = 0.3048
METER2FEET = 1 / 0.3048
DEG2RAD = 3.14159265/180
RAD2DEG = 180/3.14159265
G2FEET = 9.80665 / 0.3048
METER2MACH = 1/340.29

#########################################################################
keyboard_input = 's'

def _t2n(x):
    return x.detach().cpu().numpy()

def get_cur_date():
    year = datetime.datetime.today().year
    month = datetime.datetime.today().month if (len(str(datetime.datetime.today().month)) != 1) else "0" + str(datetime.datetime.today().month)
    day = datetime.datetime.today().day if (len(str(datetime.datetime.today().day)) != 1) else "0" + str(datetime.datetime.today().day)
    hour = datetime.datetime.today().hour if (len(str(datetime.datetime.today().hour)) != 1) else "0" + str(datetime.datetime.today().hour)
    minute = datetime.datetime.today().minute if (len(str(datetime.datetime.today().minute)) != 1) else "0" + str(datetime.datetime.today().minute)

    return str(year) + "_" + str(month) + str(day) + "_" + str(hour) + str(minute)


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
    
class SingleCombat_Test(ABC):
    def __init__(self, model_path) -> None:
        """
        model_path 는 실행파일 기준 상대 경로 "result/~~"
        
        """
        cur_path = os.path.dirname(os.path.abspath(__file__))
        model_path = cur_path + "/" + model_path
        
        args = Args()

        self.env = SingleCombatEnv("1v1/ShootMissile/vsACAM")
        self.env.seed(0)

        self.ego_policy = PPOActor(args, self.env.observation_space, self.env.action_space, device=torch.device("cuda"))
        self.ego_policy.eval()
        pt_load = torch.load(model_path)
        self.ego_policy.load_state_dict(pt_load['model_dict'])

    def run(self):
        ego_obs = self.env.reset()

        ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        masks = np.ones((1, 1))

        cur_date = get_cur_date()
        self.env.render(mode='txt', filepath = cur_date + str("_txt.acmi"))

        while True:
            ego_actions, _, ego_rnn_states = self.ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
            ego_actions = _t2n(ego_actions)
            ego_rnn_states = _t2n(ego_rnn_states)

            actions = np.concatenate((ego_actions, ego_actions), axis=0)
            ego_obs, _, dones, _ = self.env.step(actions)
            self.env.render(mode='txt', filepath = cur_date + str("_txt.acmi"))

            if dones.any():
                break

model_path = "../../../scripts/results/SingleCombat/1v1/ShootMissile/vsACAM/ppo/v1/run32/actor_-0.6395797275361561_143.pt"
test = SingleCombat_Test(model_path)
test.run()