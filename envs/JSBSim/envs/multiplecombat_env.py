import numpy as np
from typing import Tuple, Dict, Any
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str, port = 54000):
        super().__init__(config_name, port = port)
        # Env-Specific initialization here!
        self._create_records = False

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()

        # ARES 와 소켓 통신
        self.socket_send_recv(reset = True)

        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    # #@profile
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
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
                if (sim.color == "Blue"): # ONLY FOR AI
                    sim.run()

            for sim in self._tempsims.values():
                sim.run()
            
        # ARES 와 소켓 통신
        self.socket_send_recv(action)

        self.task.step(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        
        rewards = {}
        for agent_name, agent in self.agents.items():
            if (agent.color == "Blue"): # ONLY FOR AI
                reward, info = self.task.get_reward(self, agent_name, info)
                rewards[agent_name] = [reward]
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        # enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        # for enm_id in self.enm_ids:
        #     rewards[enm_id] = [enm_reward]

        dones = {}
        for agent_name, agent in self.agents.items():
            if (agent.color == "Blue"):
                done, info = self.task.get_termination(self, agent_name, info)
                dones[agent_name] = [done]
                # dones[agent_id] = [False]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info
