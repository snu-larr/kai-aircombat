import gym
from gym.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple
from ..core.simulatior import UnControlAircraftSimulator, AircraftSimulator, BaseSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config, LLA2ECEF, LLA2NEU

import socket
import math

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

        # id - state
        self.mu_id_state = {}
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
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),
                model=config.get("model", "f16"),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,
                num_missiles=config.get("missile", 0))
            
        for uid, config in self.config.rule_aircraft_configs.items():
            self._jsbsims[uid] = UnControlAircraftSimulator(
                uid=uid,
                color=config.get("color", "Blue"),
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
                elif k[0] == key[0]:
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
        
        # ARES 와 소켓 통신
        self.socket_send_recv()
        
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

        for agent_id in self.agents.keys():
            if (agent_id[0] == "A"): # ONLY FOR AI
                a_action = self.task.normalize_action(self, agent_id, action[agent_id])
                self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for agent_id, sim in self._jsbsims.items():
                if (agent_id[0] == "A"):
                    sim.run()

            for sim in self._tempsims.values():
                sim.run()

        # ARES 와 소켓 통신
        self.socket_send_recv(action)

        self.task.step(self)
        obs = self.get_obs()

        dones = {}
        for agent_id in self.agents.keys():
            if (agent_id[0] == "A"):
                done, info = self.task.get_termination(self, agent_id, info)
                dones[agent_id] = [done]

        rewards = {}
        for agent_id in self.agents.keys():
            if (agent_id[0] == "A"):
                reward, info = self.task.get_reward(self, agent_id, info)
                rewards[agent_id] = [reward]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def socket_send_recv(self, action = None):
        # 데이터 송신
        for agent_id in self.agents.keys():
            if (agent_id[0] != "R"):
                lon, lat, alt = self.agents[agent_id].get_geodetic()
                vx, vy, vz = self.agents[agent_id].get_velocity()
                x, y, z = LLA2ECEF(lon, lat, alt)

                ac_msg = "ORD|9100|6<" + agent_id + "|" + str(round(x, 6)) + "|" + str(round(y, 6)) + "|" + str(round(z, 6)) + "|" + \
                    str(round(math.sqrt(vx**2 + vy**2 + vz**2), 6)) + "|9>"

                # missile check
                if (action != None):
                    gun_trigger, aim9_trigger, aim120_trigger, chaff_trigger, flare_trigger, jammer_trigger, radar_trigger, radar_lock, target_idx = action[agent_id][4:]
                    # gun_trigger, aim9_trigger, aim120_trigger, chaff_trigger, flare_trigger, jammer_trigger, radar_trigger, radar_lock = action[agent_id][4:]
                    try:
                        target_id = [id for idx, id in enumerate(self.agents.keys()) if (idx == target_idx and id[0] == "R")][0]
                    except:
                        target_id = "X"
                else:
                    target_idx, gun_trigger, aim9_trigger, aim120_trigger = 0, 0, 0, 0
                    chaff_trigger, flare_trigger, jammer_trigger, radar_trigger, radar_lock = 0, 0, 0, 0, 0  
                    target_id = "X"

                gun_msg = "ORD|9200|3<" + agent_id + "|" + target_id + "|0>" if gun_trigger else ""
                aim9_msg = "ORD|9200|3<" + agent_id + "|" + target_id + "|1>" if aim9_trigger or target_id != "X" else ""
                aim120_msg = "ORD|9200|3<" + agent_id + "|" + target_id + "|2>" if aim120_trigger or target_id != "X" else ""
                chaff_msg = "ORD|9300|1<" + agent_id + ">" if chaff_trigger else ""
                flare_msg = "ORD|9301|1<" + agent_id + ">" if flare_trigger else ""
                    
        msg = ac_msg + gun_msg + aim9_msg + aim120_msg + chaff_msg + flare_msg
        msg = msg.encode()
        self.socket.sendall(msg)
        ###################
        # 서버로부터 action 값을 전달한 t+1초의 무장 정보와 규칙기반의 항공기의 상태 응답을 기다림
        data = self.socket.recv(self.buffer_size)
        data = data.decode()

        temp_agent_id = [id for id in self.agents.keys()][0]
        std_lon, std_lat, std_alt = self.agents[temp_agent_id].lon0, self.agents[temp_agent_id].lat0, self.agents[temp_agent_id].alt0
        
        for agent_id in self.agents.keys():
            if (agent_id[0] == "R"):
                self.agents[agent_id].recv_data = data
                self.agents[agent_id].parsing_data()
                self.agents[agent_id]._update_properties()
                mu_id_state_detected_munition_by_ai = self.agents[agent_id].mu_id_state_detected_munition_by_ai
                mu_id_target_id_dmg = self.agents[agent_id].mu_id_target_id_dmg
                ac_id_name = self.agents[agent_id].ac_id_name

        for agent_id in self.agents.keys():
            if (agent_id[0] == "A"):
                # 무장 피격 로직
                for mu_id, target_id_dmg_dict in mu_id_target_id_dmg.items():
                    for target_id, dmg in target_id_dmg_dict.items():
                        if (agent_id == ac_id_name[target_id]):
                            self.agents[agent_id].bloods -= dmg
                
                if (self.agents[agent_id].bloods < 0):
                    self.agents[agent_id].shotdown()
                else:
                    nearest_munition = []
                    nearest_dist = 999999

                    # 무장 발견 로직
                    ac_lon, ac_lat, ac_alt = self.agents[agent_id].get_geodetic()
                    for mu_id, mu_state in mu_id_state_detected_munition_by_ai.items():
                        mu_lon, mu_lat, mu_alt, mu_r, mu_p, mu_y, mu_v = mu_state

                        # print("XXX")
                        # print(str(ac_lon) + " | " + str(ac_lat) + " | " + str(ac_alt))
                        # print(str(mu_lon) + " | " + str(mu_lat) + " | " + str(mu_alt))
                        # print(str(std_lon) + " | " + str(std_lat) + " | " + str(std_alt))

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
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id in self.agents.keys() if agent_id[0] == "A"]) # ONLY FOR AI (need to train)

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id in self.agents.keys() if agent_id[0] == "A"])

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
