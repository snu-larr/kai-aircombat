from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask
from ..tasks.following_task import SAMTask
from ..tasks.sam_task import SAM_Destroy_Task

class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str, port = 4001):
        super().__init__(config_name, port = port)
        # Env-Specific initialization here!
        # assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        elif taskname == 'sam':
            self.task = SAM_Destroy_Task(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0
        self.dict_reset()

        self.socket_send_recv(reset_flag = True)
        self.reset_simulators()
        self.heading_turn_counts = 0

        # reset task
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)


    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # init_heading = self.np_random.uniform(0., 180.)
        # init_altitude = self.np_random.uniform(14000., 30000.)
        # init_velocities_u = self.np_random.uniform(400., 1200.)
        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': 0,
                'ic_h_sl_ft': 20000,
                'ic_u_fps': 800,
                # 'target_heading_deg': init_heading,
                # 'target_altitude_ft': init_altitude,
                # 'target_velocities_u_mps': init_velocities_u * 0.3048,
            })
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()
