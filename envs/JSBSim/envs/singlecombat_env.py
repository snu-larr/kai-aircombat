import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str, port = 54000):
        super().__init__(config_name, port = port)
        # Env-Specific initialization here!
        # assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        
        # id - name (초기 setting)
        self.ac_id_name, self.ac_name_id, self.ac_id_iff = {}, {}, {}
        self.ed_id_name, self.ed_name_id, self.ed_id_upid = {}, {}, {}
        self.mu_id_name, self.mu_name_id, self.mu_id_upid = {}, {}, {}
        self.sam_id_name = {}

        # aircraft/munition id - state
        self.ac_id_state = {}
        self.mu_id_state = {}
        self.sam_id_state = {}
        
        # 전자장비 id - state
        self.rad_id_state, self.rwr_id_state, self.mws_id_state = {}, {}, {}

        # damage page
        self.mu_id_target_id_dmg = {}

        # detected data
        self.ac_id_state_detected_by_ai, self.mu_id_state_detected_by_ai = {}, {}
        ###

        # TODO : reset trigger 전달 및 초기값 수신 이후, 객체 생성 및 reload 진행
        self.socket_send_recv(reset_flag = True)
        self.reset_simulators()

        # reset task
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # self.init_states[0].update({
        #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
        #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        # })
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()
