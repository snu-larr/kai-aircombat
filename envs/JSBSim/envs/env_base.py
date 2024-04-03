import gym
from gym.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from ..core.simulatior import UnControlAircraftSimulator, AircraftSimulator, BaseSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config, LLA2ECEF, LLA2NEU

import socket
import math
import traceback
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

    def __init__(self, config_name: str, server_ip = "127.0.0.1", port = 54000, buffer_size = 1024):
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
        self.socket.connect((self.server_ip, self.port))

        # id - name (초기 setting)
        self.ac_id_name, self.ac_name_id = {}, {}
        self.ed_id_name, self.ed_name_id, self.ed_id_upid = {}, {}, {}
        self.mu_id_name, self.mu_name_id, self.mu_id_upid = {}, {}, {}

        # aircraft/munition id - state
        self.ac_id_state = {}
        self.mu_id_state = {}
        
        # 전자장비 id - state
        self.rad_id_state, self.rwr_id_state, self.mws_id_state = {}, {}, {}

        # damage page
        self.mu_id_target_id_dmg = {}

        # detected data
        self.ac_id_state_detected_by_ai, self.mu_id_state_detected_by_ai = {}, {}
        ###

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
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        self.load_task()
        self.load_simulator()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)

    def load_simulator(self):
        self._jsbsims = {}     # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.ai_aircraft_configs.items():
            # AI
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Blue"),
                model=config.get("model", "f16"),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,
                num_missiles=config.get("missile", 0))
            
        for uid, config in self.config.rule_aircraft_configs.items():
            # Rule
            self._jsbsims[uid] = UnControlAircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),
                model=config.get("model", "f16"),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,
                num_missiles=config.get("missile", 0))
            
        # Different teams have different uid[0]
        _default_team_uid = list(self._jsbsims.keys())[0][0]
        self.ego_ids = [uid for uid in self._jsbsims.keys() if uid[0] == _default_team_uid]
        self.enm_ids = [uid for uid in self._jsbsims.keys() if uid[0] != _default_team_uid]

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

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (np.ndarray): initial observation
        """
        # reset sim
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()
        
        # state reset
        self.ac_id_name = {}
        self.mu_id_state = {}
        self.ac_id_state_detected_by_ai = {}

        # ARES 와 소켓 통신
        self.socket_send_recv(reset = True)
        
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

        for agent_name, agent in self.agents.items():
            if (agent.color == "Blue"): # ONLY FOR AI
                a_action = self.task.normalize_action(self, agent_name, action[agent_name])
                self.agents[agent_name].set_property_values(self.task.action_var, a_action)
        
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for agent_name, sim in self._jsbsims.items():
                if (sim.color == "Blue"):
                    sim.run()

            for sim in self._tempsims.values():
                sim.run()

        # ARES 와 소켓 통신
        self.socket_send_recv(action)

        self.task.step(self)
        obs = self.get_obs()

        dones = {}
        for agent_name, agent in self.agents.items():
            if (agent.color == "Blue"):
                done, info = self.task.get_termination(self, agent_name, info)
                dones[agent_name] = [done]

        rewards = {}
        for agent_name, agent in self.agents.items():
            if (agent.color == "Blue"):
                reward, info = self.task.get_reward(self, agent_name, info)
                rewards[agent_name] = [reward]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def parsing_data(self, data):
        packet_datas = data.split("/")
        
        for packet_data in packet_datas:
            header, id, data = packet_data.split("|", 2)
            cnt = int(data.split("<")[0])
            data = re.search(r'\<(.*?)\>', data).group(1)

            if (id == "7011"): # 항공기 설정
                ac_id, ac_name, iff = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.ac_id_name[ac_id] = ac_name
                self.ac_name_id[ac_name] = ac_id

            if (id == "7015"): # 전자장비 설정
                ed_id, ed_name, iff, upid = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.ed_id_name[ed_id] = ed_name
                self.ed_name_id[ed_name] = ed_id
                self.ed_id_upid[ed_id] = upid

            if (id == "7016"): # 무장 설정
                mu_id, mu_name, iff, upid = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.mu_id_name[mu_id] = mu_name
                self.mu_name_id[mu_name] = mu_id
                self.mu_id_upid[mu_id] = upid 

            if (id == "7101"): # 항공기 기동
                id, lon, lat, alt, r, p, y, vn, ve, vd, vbx, vby, vbz, vc, an, ae, ad = [float(x) if x.replace('.', '', 1).isdigit() else x for x in data.split("|")]
                self.ac_id_state[id] = [float(lon), float(lat), float(alt), float(r), float(p), float(y), float(vn), float(ve), float(vd), float(vbx), float(vby), float(vbz), float(vc), float(an), float(ae), float(ad)]

            if (id == "7102"): # 미사일 기동
                mu_id, lon, lat, alt, r, p, y, v = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.mu_id_state[mu_id] = [lon, lat, alt, r, p, y, v] 

            if (id == "7201"): # 레이더 탐지
                ed_id, target_id = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.rad_id_state[ed_id] = target_id

            if (id == "7202"): # RWR 
                ed_id, target_id = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.rwr_id_state[ed_id] = target_id

            if (id == "7203"): # MWS
                ed_id, target_id = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.mws_id_state[ed_id] = target_id

            if (id == "7401"): # 미사일 피격
                mu_id, target_id, dmg = [float(x) if x.replace("-", "").replace('.', '').isdigit() else x for x in data.split("|")]
                self.mu_id_target_id_dmg[mu_id] = {**self.mu_id_target_id_dmg, **{target_id: dmg}}

        ################################
        for ed_id, target_id in self.mws_id_state.items():
            try:
            # AI 가 가지고 있는 MWS 로 무장 정보를 얻게 되면 해당 값을 공유
                if (self.agents[self.ac_id_name[self.ed_id_upid[ed_id]]].color == "Red"):
                
                    lon, lat, alt, r, p, y, v = self.mu_id_state[target_id]
                    self.mu_id_state_detected_by_ai[target_id] = [lon, lat, alt, r, p, y, v]
            except:
                # print(Exception, err)
                # print(traceback.format_exc())
                # MWS 에 잡힌 target id 가 munition 이 아닌 경우?
                pass
        
        for ed_id, target_id in self.rwr_id_state.items():
            # AI 가 가지고 있는 RWR 로 무장 혹은 항공기 정보를 얻게 되면 해당 값을 공유
            try:
                if (self.agents[self.ac_id_name[self.ed_id_upid[ed_id]]].color == "Red"):
                    agent = self.agents[self.ac_id_name[self.ed_id_upid[ed_id]]]
                
                    if (target_id in self.mu_id_state.keys()):
                        lon, lat, alt, r, p, y, v = self.mu_id_state[target_id]
                        self.mu_id_state_detected_by_ai[target_id] = [lon, lat, alt, r, p, y, v]
                    
                    if (target_id in self.ac_id_state.keys()):
                        lon, lat, alt, r, p, y, vn, ve, vd, vbx, vby, vbz, vc, an, ae, ad = self.ac_id_state[target_id]
                        self.ac_id_state_detected_by_ai[target_id] = [lon, lat, alt, r, p, y, vn, ve, vd, vbx, vby, vbz, vc, an, ae, ad]
                
            except Exception as err:
                # print(Exception, err)
                # print(traceback.format_exc())
                pass

        ################################
        for ed_id, target_id in self.rad_id_state.items():
            # AI 가 가지고 있는 radar 로 상대 (규칙기반) 정보를 얻게 되면 해당 값을 공유
            try:
                if (self.agents[self.ac_id_name[self.ed_id_upid[ed_id]]].color == "Red"):
                    lon, lat, alt, r, p, y, vn, ve, vd, vbx, vby, vbz, vc, an, ae, ad = self.ac_id_state[target_id]
                    self.ac_id_state_detected_by_ai[target_id] = [lon, lat, alt, r, p, y, vn, ve, vd, vbx, vby, vbz, vc, an, ae, ad]
            except Exception as err:
                # RAD에 잡힌 target id 가 aircraft 가 아닌 경우?
                # print(Exception, err)
                # print(traceback.format_exc())
                pass

        ################################
        # 피격 판정 중 자신이 맞았다면 반영
        for mu_id, tgt_id_dmg_dict in self.mu_id_target_id_dmg.items():
            for tgt_id, dmg in tgt_id_dmg_dict.items():
                for agent_name, agent in self.agents.items():
                    agent_id = self.ac_name_id[agent_name]
                    if (tgt_id == agent_id):
                        agent.bloods -= dmg    
        
        for agent_name, agent in self.agents.items():
            if (agent.bloods <= 0):
                agent.shotdown()
        ################################

    def socket_send_recv(self, action = None, reset = False):
        # 데이터 송신
        msg = ""

        # target idx 와 aircraft id 의 matching
        target_idx_ac_id = {}
        for idx, ac_id in enumerate(self.ac_id_name.keys()):
            target_idx_ac_id[idx] = ac_id

        for agent_name, agent in self.agents.items():
            if (agent.color != "Red"):
                lon, lat, alt = self.agents[agent_name].get_geodetic()
                vx, vy, vz = self.agents[agent_name].get_velocity()
                x, y, z = LLA2ECEF(lon, lat, alt)

                ac_msg = "ORD|9100|6<" + agent_name + "|" + str(round(x, 6)) + "|" + str(round(y, 6)) + "|" + str(round(z, 6)) + "|" + \
                    str(round(math.sqrt(vx**2 + vy**2 + vz**2), 6)) + "|9>"

                # missile check
                if (action != None):
                    # target idx 는 ARES 에서 주는 항공기 정보를 바탕으로 idx 부여
                    gun_trigger, aim9_trigger, aim120_trigger, chaff_flare_trigger, jammer_trigger, radar_trigger, target_idx = action[agent_name][4:]
                    
                    # radar locking 이 안된경우에는 target idx 값이 무의미하도록 변경 필요, ex) [target_idx : 3 / target_id : R0001]
                    detected_ac_list = [target_id for rad_id, target_id in self.rad_id_state.items() if agent_name == self.ac_id_name[self.ed_id_upid[rad_id]]]
                    if (target_idx_ac_id[target_idx] in detected_ac_list):
                        target_id = self.ac_id_name[target_idx_ac_id[target_idx]]
                    else:
                        target_id = "X"

                    # RWR 이 울린 경우에만 Chaff/Flare 를 발사하도록 변경
                    detected_rwr_list = [target_id for rwr_id, target_id in self.rwr_id_state.items() if agent_name == self.ac_id_name[self.ed_id_upid[rwr_id]]]
                    if (len(detected_rwr_list) == 0 and chaff_flare_trigger > 0):
                        chaff_flare_trigger = 0

                else:
                    target_idx, gun_trigger, aim9_trigger, aim120_trigger = 0, 0, 0, 0
                    chaff_flare_trigger, jammer_trigger, radar_trigger = 0, 0, 0
                    target_id = "X"

                gun_msg = "ORD|9200|3<" + agent_name + "|" + target_id + "|0>" if gun_trigger else ""
                aim9_msg = "ORD|9200|3<" + agent_name + "|" + target_id + "|1>" if aim9_trigger or target_id != "X" else ""
                aim120_msg = "ORD|9200|3<" + agent_name + "|" + target_id + "|2>" if aim120_trigger or target_id != "X" else ""
                chaff_flare_msg = "ORD|9300|1<" + agent_name + ">" if chaff_flare_trigger else ""
                msg += ac_msg + gun_msg + aim9_msg + aim120_msg + chaff_flare_msg

        reset_msg = "ORD|9400" if (reset) else ""
        
        ##### reset 시 callstack 확인
        # if (reset):
        #     for line in traceback.format_stack():
        #         print(line.strip())

        msg += reset_msg

        msg = msg.encode()
        self.socket.sendall(msg)
        ###################
        # 서버로부터 action 값을 전달한 t+1초의 무장 정보와 규칙기반의 항공기의 상태 응답을 기다림
        data = self.socket.recv(self.buffer_size)
        data = data.decode()

        temp_agent_id = [id for id in self.agents.keys()][0]
        std_lon, std_lat, std_alt = self.agents[temp_agent_id].lon0, self.agents[temp_agent_id].lat0, self.agents[temp_agent_id].alt0
        
        # 무장 정보
        self.parsing_data(data)
        
        for agent_name, agent in self.agents.items():
            if (agent.color == "Red"):
                agent_id = [id for name, id in self.ac_name_id.items() if name == agent_name][0]

                agent.set_ac_state(self.ac_id_state[agent_id], agent_id)
                agent._update_properties(agent_id)

        for agent_name, agent in self.agents.items():
            if (agent.color == "Blue"):
                # 무장 피격 로직
                for mu_id, target_id_dmg_dict in self.mu_id_target_id_dmg.items():
                    for target_id, dmg in target_id_dmg_dict.items():
                        if (agent_name == self.ac_id_name[target_id]):
                            agent.bloods -= dmg
                
                if (agent.bloods < 0):
                    agent.shotdown()
                else:
                    nearest_munition = []
                    nearest_dist = 999999

                    # 무장 발견 로직
                    ac_lon, ac_lat, ac_alt = agent.get_geodetic()
                    for mu_id, mu_state in self.mu_id_state_detected_by_ai.items():
                        mu_lon, mu_lat, mu_alt, mu_r, mu_p, mu_y, mu_v = mu_state

                        # deg deg m -> m m m
                        ego_position = LLA2NEU(ac_lon, ac_lat, ac_alt, std_lon, std_lat, std_alt)
                        enm_position = LLA2NEU(mu_lon, mu_lat, mu_alt, std_lon, std_lat, std_alt)

                        dist = math.sqrt((ego_position[0] - enm_position[0]) ** 2 + (ego_position[1] - enm_position[1]) ** 2 + (ego_position[2] - enm_position[2]) ** 2)
                        if (dist < nearest_dist):
                            nearest_munition = [mu_lon, mu_lat, mu_alt, mu_r, mu_p, mu_y, mu_v]

                    # ac 에 가장 가까운 munition 을 선택하고, 해당 미사일을 ac의 nearest_munition 변수에 추가
                    if (len(nearest_munition) > 0):
                        self.agents[agent_name].nearest_munition = nearest_munition
                    

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return dict([(agent_name, self.task.get_obs(self, agent_name)) for agent_name, agent in self.agents.items() if agent.color == "Blue"]) # ONLY FOR AI (need to train)

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_name) for agent_name in self.agents.keys()])
        return dict([(agent_name, state.copy()) for agent_name, agent in self.agents.keys() if agent.color == "Blue"])

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
                for air_id, sim in self._jsbsims.items():
                    if (air_id == aircraft_id or aircraft_id == None):    
                        log_msg = sim.log()
                        if log_msg is not None:
                            f.write(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
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
