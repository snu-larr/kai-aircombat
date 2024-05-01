import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from ..core.simulatior import UnControlAircraftSimulator, AircraftSimulator, BaseSimulator, UnControlSAMSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config, LLA2ECEF, LLA2NEU

import socket
import math
import traceback
import time
import re

class BaseEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An BaseEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "txt"]}

    # 192.168.100.111
    # 192.168.100.33
    # 127.0.0.1
    def __init__(self, config_name: str, server_ip = "127.0.0.1", port = 4001, buffer_size = 65000):
        # basic args
        self.config = parse_config(config_name)
        self.max_steps = getattr(self.config, 'max_steps', 100)  # type: int
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))
        self._create_records = False
        self.acmi_file_path = None
        self.render_step = 0

        ###
        # socket comm
        self.server_ip = server_ip
        self.port = port
        self.buffer_size = buffer_size

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print("IP / PORT : ", self.server_ip, self.port)
        self.socket.connect((self.server_ip, self.port))

        self.ed_id_upid = {}
        self.mu_id_upid = {}

        # aircraft/munition id - state
        self.ac_id_state = {}
        self.mu_id_state = {}
        self.sam_id_state = {}
        
        # 전자장비 id - state
        self.rad_upid_state, self.rwr_id_state, self.mws_id_state = {}, {}, {}

        # damage page
        self.mu_id_target_id_dmg = {}

        self.first_exe_flag = True

        self._jsbsims = {}     # type: Dict[str, AircraftSimulator]
        self._samsims = {}
        self.load()

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def num_ai_agents(self) -> int:
        return self.task.num_ai_agents

    @property
    def observation_space(self) -> gym.Space:
        return self.task.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self._jsbsims

    @property
    def sams(self) -> Dict[str, UnControlSAMSimulator]:
        return self._samsims

    @property
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        self.load_task()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)

    def load_simulator_ACAM(self):
        self._jsbsims = {}
        self._samsims = {}

        for ac_id, state in self.ac_id_init_state.items():
            lon, lat, alt = state
            iff = self.ac_id_iff[ac_id]
            op_mode = self.ac_id_opmode[ac_id]

            mu_id = [id for id, up_id in self.mu_id_upid.items() if up_id == ac_id]

            if (op_mode == 1): # 0 : Rule / 1 : AI / 2 : SIM
                temp_color = "Blue" if iff == 1 else "Red"
                self._jsbsims[ac_id] = AircraftSimulator(
                    uid = ac_id,
                    color = temp_color,
                    model = "f16",
                    init_state = {
                        "ic_h_sl_ft": alt * 1 / 0.3048,
                        "ic_lat_geod_deg": lat,
                        "ic_long_gc_deg": lon,
                        "ic_psi_true_deg" : 0.0,
                        "ic_u_fps": 800.0, # 초기속도
                    },
                    origin = [124.00, 37.00, 0.0],
                    sim_freq = self.sim_freq,
                    num_missiles = len(mu_id)
                )

        for ac_id, state in self.ac_id_init_state.items():
            lon, lat, alt = state
            iff = self.ac_id_iff[ac_id]
            op_mode = self.ac_id_opmode[ac_id]

            mu_id = [id for id, up_id in self.mu_id_upid.items() if up_id == ac_id]

            if (op_mode != 1): # 0 : Rule / 1 : AI / 2 : SIM
                temp_color = "Blue" if iff == 1 else "Red"

                self._jsbsims[ac_id] = UnControlAircraftSimulator(
                    uid = ac_id,
                    color = temp_color,
                    model = "f16",
                    init_state = {
                        "ic_h_sl_ft": alt * 1 / 0.3048,
                        "ic_lat_geod_deg": lat,
                        "ic_long_gc_deg": lon,
                        "ic_psi_true_deg" : 0,
                        "ic_u_fps": 0,
                    },
                    origin = [124.00, 37.00, 0.0],
                    sim_freq = self.sim_freq,
                    num_missiles = len(mu_id)
                )

        for sam_id, state in self.sam_id_state.items():
            lon, lat, alt = state
            
            self._samsims[sam_id] = UnControlSAMSimulator(
                uid = sam_id,
                color = "Red",
                model = "SAM",
                init_state = {
                    "ic_h_sl_ft": alt * 1 / 0.3048,
                    "ic_lat_geod_deg": lat,
                    "ic_long_gc_deg": lon,
                },
                origin = [124.00, 37.00, 0.0],
                sim_freq = self.sim_freq
            )

        # EGO : Blue / ENM : Red
        # 모든 AI 는 Blue 라고 가정함
        self.ego_ids = [uid for uid, agent in self._jsbsims.items() if agent.color == "Blue"]
        self.enm_ids = [uid for uid, agent in self._jsbsims.items() if agent.color == "Red"]

        # Link jsbsims
        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass
                elif s.color == sim.color:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

        self._tempsims = {}    # type: Dict[str, BaseSimulator]

    def add_temp_simulator(self, sim: BaseSimulator):
        self._tempsims[sim.uid] = sim

    def dict_reset(self):
        self.ac_id_iff = {}
        self.ac_id_opmode = {}
        self.ed_id_upid = {}
        self.mu_id_upid = {}
        
        # aircraft/munition id - state
        self.ac_id_state = {}
        self.ac_id_init_state = {}
        self.mu_id_state = {}
        self.sam_id_state = {}
        self.ed_id_state = {}
        self.hmd_id_state = {}

        # id - spec
        self.launcher_id_spec = {}
        self.rad_id_spec = {}
        self.jammer_id_spec = {}

        # 전자장비 id - state
        self.rad_upid_state, self.rwr_id_state, self.mws_id_state = {}, {}, {}

        # damage page
        self.mu_id_target_id_dmg = {}
        ###

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (np.ndarray): initial observation
        """
        self.dict_reset()
        self.socket_send_recv(reset_flag = True)

        # reset sim
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()
        
        self._tempsims.clear()
        # reset task
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        
        # apply actions
        action = self._unpack(action)

        for agent_id, agent in self.agents.items():
            if (agent.mode == "AI"): # ONLY FOR AI
                a_action = self.task.normalize_action(self, agent_id, action[agent_id])
                self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for agent_id, sim in self._jsbsims.items():
                if (sim.mode == "AI"):
                    sim.run()

            for sim in self._tempsims.values():
                sim.run()

        # ARES 와 소켓 통신
        self.socket_send_recv(action)

        self.task.step(self)
        obs = self.get_obs()

        dones = {}
        for agent_id, agent in self.agents.items():
            if (agent.mode == "AI"):
                done, info = self.task.get_termination(self, agent_id, info)
                dones[agent_id] = [done]

        rewards = {}
        for agent_id, agent in self.agents.items():
            if (agent.mode == "AI"):
                reward, info = self.task.get_reward(self, agent_id, info)
                rewards[agent_id] = [reward]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def parsing_data(self, data, reset_flag = False):
        # 입력에 "/" 구분자가 없는 경우
        packet_datas = data.split("ORD")[1:]
        
        self.mu_id_target_id_dmg = {}
        for packet_data in packet_datas:
            id, data = packet_data.split("|", 2)[1:]
            cnt = int(data.split("<")[0])
            data = re.search(r'\<(.*?)\>', data).group(1)
        #################
        
            if (id == "7011"): # 항공기 설정
                ac_id, iff, up_id, obj_type, lon, lat, alt, op_mode = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                
                self.ac_id_iff[ac_id] = float(iff)
                self.ac_id_opmode[ac_id] = float(op_mode)
                self.ac_id_init_state[ac_id] = [float(lon), float(lat), float(alt)]

            if (id == "7013"): # 지대공 위협 설정
                sam_id, iff, upid, obj_type, lon, lat, alt, op_dir = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                
                self.sam_id_state[sam_id] = [lon, lat, alt]

            if (id == "7015"): # 전자장비 설정
                ed_id, iff, upid, obj_type = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                
                self.ed_id_upid[ed_id] = upid
                self.ed_id_state[ed_id] = [iff, upid, obj_type]

            if (id == "7016"): # 무장 설정
                mu_id, iff, upid, obj_type = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                
                self.mu_id_upid[mu_id] = upid 

            if (id == "7021"): # 발사대 범위
                launcher_id, max_elev, min_elev, max_dist, min_dist = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.launcher_id_spec[launcher_id] = [min_elev, max_elev, min_dist, max_dist]

            if (id == "7022"): # 레이더 범위
                rad_id, direction, azimuth, max_elev, min_elev, max_dist = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.rad_id_spec[rad_id] = [direction, azimuth, min_elev, max_elev, max_dist]

            if (id == "7023"): # 지상 재머
                jammer_id, direction, azimuth, max_elev, min_elev, max_dist, min_dist = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.jammer_id_spec[jammer_id] = [direction, azimuth, min_elev, max_elev, min_dist, max_dist]

            if (id == "8101"): # 항공기 기동
                # time, id, lon, lat, alt, r, p, y, vn, ve, vu, vbx, vby, vbz, vc, G, remain_fuel, weight, thrust, distance_to_target, AA_to_target, RPM
                ac_state = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                time, id, lon, lat, alt, r, p, y, vn, ve, vu, vbx, vby, vbz, vc, G, remain_fuel, weight, thrust, distance_to_target, AA_to_target, RPM = [0 if x == "1.#QNAN0" or x == "1.#INF00" else x for x in ac_state]
                
                # TODO : 아래 id 삭제하기 & inf/nan 처리 필요
                if (id != 110200001):
                    self.ac_id_state[id] = [float(lon), float(lat), float(alt), float(r), float(p), float(y), float(vn), float(ve), float(-vu), float(vbx), float(vby), float(vbz), float(vc)]

            if (id == "8102"): # 미사일 기동
                time, mu_id, lon, lat, alt, r, p, y, v = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.mu_id_state[mu_id] = [lon, lat, alt, r, p, y, v] 

            if (id == "8201"): # 레이더 탐지 : 해당 레이더 탐지는 적군기 Aircraft 만을 탐지함 (적 Missile X)
                time, ac_id, obj_type, rad_id, target_id, angle, dist = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.rad_upid_state[ac_id] = target_id

            if (id == "8202"): # RWR : 해당 RWR 은 적 무장만을 탐지함 (적 Aircraft X)
                time, ed_id, target_id, target_rad_id, target_code, angle, dist = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.rwr_id_state[ed_id] = target_id

            if (id == "8204"): # 육안탐지
                time, ac_id, target_id, angle, dist = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.hmd_id_state[ac_id] = target_id

            if (id == "8401"): # 미사일 피격
                time, ac_id, mu_id, target_id, target_lon, target_lat, target_alt, dmg, cum_dmg = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.mu_id_target_id_dmg[mu_id] = {**self.mu_id_target_id_dmg, **{target_id: dmg}}

            if (id == "8403"): # AAA 피격
                time, mu_id, target_id, target_lon, target_lat, target_alt, dmg = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.mu_id_target_id_dmg[mu_id] = {**self.mu_id_target_id_dmg, **{target_id: dmg}}

        ###############################
        # 필요 Simulator load & reset
        if (reset_flag):
            if (len(self.agents) == 0):
                # TODO : load ACAM 을 할때, 각 항공기의 잔여 무장 개수 확인 필요
                self.load_simulator_ACAM()
            else:
                for agent_id, agent in self.agents.items():
                    # lon, lat, alt, r, p, y, vn, ve, vd, vbx, vby, vbz, vc = self.ac_id_state[agent_id]
                    lon, lat, alt = self.ac_id_init_state[ac_id]

                    new_state = {
                        "ic_h_sl_ft": alt * 1 / 0.3048,
                        "ic_lat_geod_deg": lat,
                        "ic_long_gc_deg": lon,
                        "ic_psi_true_deg": 0,
                        "ic_u_fps": 0,
                    }
                    agent.reload(new_state = new_state)

        # 피격 판정 중 자신이 맞았다면 반영
        for mu_id, tgt_id_dmg_dict in self.mu_id_target_id_dmg.items():
            for tgt_id, dmg in tgt_id_dmg_dict.items():
                for agent_id, agent in self.agents.items():
                    if (tgt_id == agent_id):
                        agent.bloods -= dmg    
                
                for sam_id, sam in self.sams.items():
                    if (tgt_id == sam_id):
                        sam.bloods -= dmg

        for _, agent in self.agents.items():
            if (agent.bloods <= 0):
                agent.shotdown()

        for _, sam in self.sams.items():
            if (sam.bloods <= 0):
                sam.shotdown()
        ################################

    def socket_send_recv(self, action = None, reset_flag = False):
        # 데이터 송신
        msg = ""

        if (self.first_exe_flag == False):
            if (reset_flag):
                msg = "ORD|9900"
            else:
                # 학습 대상이 아닌 다른 항공기의 indexing dict
                target_idx_ac_id = {}
                for idx, ac_id in enumerate(self.enm_ids):
                    target_idx_ac_id[idx] = ac_id

                for agent_id, agent in self.agents.items():
                    if (agent.mode == "AI"):

                        lon, lat, alt = self.agents[agent_id].get_geodetic()
                        roll, pitch, yaw = self.agents[agent_id].get_rpy()
                        vx, vy, vz = self.agents[agent_id].get_velocity()
                        # x, y, z = LLA2ECEF(lon, lat, alt)

                        ac_msg = "ORD|9100|9<" + str(agent_id) + "|" + str(round(lat, 6)) + "|" + str(round(lon, 6)) + "|" + str(round(alt, 6)) + "|" + \
                            str(round(roll, 6)) + "|" + str(round(pitch, 6)) + "|" + str(round(yaw, 6))  + "|" + \
                            str(round(math.sqrt(vx**2 + vy**2 + vz**2), 6)) + "|10000>"

                        # missile check
                        if (action != None and len(action[agent_id]) == 11):
                            # target idx 는 ARES 에서 주는 항공기 정보를 바탕으로 idx 부여
                            gun_trigger, aim9_trigger, aim120_trigger, chaff_flare_trigger, jammer_trigger, radar_trigger, target_idx = action[agent_id][4:]
                            
                            detected_ac_list = [target_id for ac_id, target_id in self.rad_upid_state.items() if agent_id == ac_id]
                            if (target_idx_ac_id[target_idx] in detected_ac_list):
                                target_id = str(target_idx_ac_id[target_idx])
                            else:
                                target_id = "X"

                            # RWR 이 울린 경우에만 Chaff/Flare 를 발사하도록 변경
                            detected_rwr_list = [target_id for rwr_id, target_id in self.rwr_id_state.items() if agent_id == self.ed_id_upid[rwr_id]]
                            if (len(detected_rwr_list) == 0 and chaff_flare_trigger > 0):
                                chaff_flare_trigger = 0

                        elif (len(action[agent_id]) == 7):
                            gun_trigger, aim9_trigger, aim120_trigger = action[agent_id][4:]
                            chaff_flare_trigger, jammer_trigger, radar_trigger = 0, 0, 0
                            target_id = [str(id) for id in self.sams.keys()][0]

                        else:
                            target_idx, gun_trigger, aim9_trigger, aim120_trigger = 0, 0, 0, 0
                            chaff_flare_trigger, jammer_trigger, radar_trigger = 0, 0, 0
                            target_id = "X"

                        # ed_state = [iff, upid, obj_type]
                        # TODO : 507??
                        jammer_id_list = [ed_id for ed_id, ed_state in self.ed_id_state.items() if ed_state[2] == 507 and ed_state[1] == agent_id]
                        jammer_id = jammer_id_list[0] if (len(jammer_id_list) != 0) else ""
                        
                        # 9200 / <agent_id, target_id, munition_id>
                        # TODO : 0, 1, 2 를 munition_id 로 수정해야함
                        gun_msg = "ORD|9200|3<" + str(agent_id) + "|" + str(target_id) + "|0>" if gun_trigger and target_id != "X" else ""
                        aim9_msg = "ORD|9200|3<" + str(agent_id) + "|" + str(target_id) + "|1>" if aim9_trigger and target_id != "X" else ""
                        aim120_msg = "ORD|9200|3<" + str(agent_id) + "|" + str(target_id) + "|2>" if aim120_trigger and target_id != "X" else ""
                        chaff_msg = "ORD|9300|1<" + str(agent_id) + ">" if chaff_flare_trigger else ""
                        flare_msg = "ORD|9301|1<" + str(agent_id) + ">" if chaff_flare_trigger else ""

                        jammer_msg = ""
                        if (jammer_id != ""):
                            jammer_msg = "ORD|9401|3<" + str(agent_id) + "|" + str(jammer_id) + "|"
                            
                            if (jammer_trigger):
                                jammer_msg += "1>"
                            else:
                                jammer_msg += "0>"

                        msg += ac_msg + gun_msg + aim9_msg + aim120_msg + chaff_msg + flare_msg + jammer_msg

            ##### reset 시 callstack 확인
            # if (reset):
            #     for line in traceback.format_stack():
            #         print(line.strip())

            # print("SEND DATA : ", msg)

            msg = msg.encode()
            self.socket.sendall(msg)
            
        ###################
        # 서버로부터 action 값을 전달한 t+1초의 무장 정보와 규칙기반의 항공기의 상태 응답을 기다림
        
        data = self.socket.recv(self.buffer_size)
        data = data.decode('cp949')

        # 7000번만 들어올 경우에만 통과
        # CHECK : cpp test code 에서는 주석필요
        # if (reset_flag):
        #     while True:
        #         packet_datas = data.split("ORD")[1:]
        #         id_list = [packet_data.split("|", 2)[1] for packet_data in packet_datas]
        #         if (any([id[0] == "8" for id in id_list])):
        #             print("###### THERE IS EIGHT NUMBER!")

        #             data = self.socket.recv(self.buffer_size)
        #             data = data.decode('cp949')
        #             continue
        #         break

        # print(data)
        # print(len(data))

        # 무장 정보
        self.parsing_data(data, reset_flag = reset_flag)
        
        # first flag setting (처음 recv 후 False 로 변경)
        self.first_exe_flag = False

        temp_agent_id = [id for id in self.agents.keys()][0]
        std_lon, std_lat, std_alt = self.agents[temp_agent_id].lon0, self.agents[temp_agent_id].lat0, self.agents[temp_agent_id].alt0
        
        if (reset_flag):
            for sam_id, sam in self.sams.items():
                sam.set_sam_state(self.sam_id_state[sam_id], sam_id)
                sam._update_properties(sam_id)


        for agent_id, agent in self.agents.items():
            if (agent.mode == "Rule"):
                
                if (len(self.ac_id_state) == 0):
                    agent.set_ac_lla(self.ac_id_init_state[agent_id], agent_id)
                    agent._update_lla_properties(agent_id)

                else:
                    agent.set_ac_state(self.ac_id_state[agent_id], agent_id)
                    agent._update_properties(agent_id)
                

        for agent_id, agent in self.agents.items():
            if (agent.mode == "AI"):
                # 가장 가까운 무장 갱신
                nearest_munition = []
                nearest_dist = 999999

                ac_lon, ac_lat, ac_alt = agent.get_geodetic()
                for ac_id, mu_id in self.rwr_id_state.items():
                    if (agent_id != ac_id):
                        continue

                    mu_lon, mu_lat, mu_alt, mu_r, mu_p, mu_y, mu_v = self.mu_id_state[mu_id]

                    # deg deg m -> m m m
                    ego_position = LLA2NEU(ac_lon, ac_lat, ac_alt, std_lon, std_lat, std_alt)
                    enm_position = LLA2NEU(mu_lon, mu_lat, mu_alt, std_lon, std_lat, std_alt)

                    dist = math.sqrt((ego_position[0] - enm_position[0]) ** 2 + (ego_position[1] - enm_position[1]) ** 2 + (ego_position[2] - enm_position[2]) ** 2)
                    if (dist < nearest_dist):
                        nearest_munition = [mu_lon, mu_lat, mu_alt, mu_r, mu_p, mu_y, mu_v]

                # ac 에 가장 가까운 munition 을 선택하고, 해당 미사일을 ac의 nearest_munition 변수에 추가
                if (len(nearest_munition) > 0):
                    self.agents[agent_id].nearest_munition = nearest_munition
                

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id, agent in self.agents.items() if agent.mode == "AI"]) # ONLY FOR AI (need to train)

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id, agent in self.agents.items() if agent.mode == "AI"])


    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._tempsims.values():
            sim.close()
        self._jsbsims.clear()
        self._tempsims.clear()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi', aircraft_id = None):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - txt: output to txt.acmi files

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        if mode == "txt":
            if (self.acmi_file_path != filepath):
            # if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                # self._create_records = True
                self.acmi_file_path = filepath
                self.render_step = 0
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.render_step * self.time_interval
                self.render_step += 1
                f.write(f"#{timestamp:.3f}\n")

                # aircraft
                for air_id, sim in self._jsbsims.items():
                    if (air_id == aircraft_id or aircraft_id == None):    
                        log_msg = sim.log()
                        if log_msg is not None:
                            f.write(log_msg + "\n")
                
                # temp??
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")

                # sam
                for sam_id, sam in self._samsims.items():
                    log_msg = sam.log()
                    if (log_msg is not None):
                        f.write(log_msg + "\n")

        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack seperated key-value dict into grouped np.ndarray"""
        ego_data = np.array([data[uid] for uid in self.ego_ids])
        
        try:
            enm_data = np.array([data[uid] for uid in self.enm_ids])
        except:
            enm_data = np.array([])
        
        if enm_data.shape[0] > 0:
            data = np.concatenate((ego_data, enm_data))  # type: np.ndarray
        else:
            data = ego_data  # type: np.ndarray
        try:
            assert np.isnan(data).sum() == 0
        except AssertionError:
            data = np.nan_to_num(data)

            # import pdb
            # pdb.set_trace()
        # only return data that belongs to RL agents
        try:
            ret = data[:self.num_agents, ...]
        except Exception as e:
            print("e : " , e)

        return ret

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack grouped np.ndarray into seperated key-value dict"""
        # assert isinstance(data, (np.ndarray, list, tuple)) and len(data) == self.num_agents
        # unpack data in the same order to packing process
        unpack_data = dict(zip((self.ego_ids + self.enm_ids)[:len(data)], data))
        # fill in None for other not-RL agents
        # for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
        #     unpack_data[agent_id] = None
        return unpack_data
