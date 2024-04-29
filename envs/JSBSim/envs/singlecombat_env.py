import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask
from ..tasks.sam_task import SAM_Destroy_Task
from ..tasks.heading_task import HeadingTask

class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str, port = 4001):
        super().__init__(config_name, port = port)
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'sam':
            self.task = SAM_Destroy_Task(self.config)
        elif taskname == "heading":
            self.task = HeadingTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.dict_reset()

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

        # TODO : 초기값 랜덤 주입?
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()
