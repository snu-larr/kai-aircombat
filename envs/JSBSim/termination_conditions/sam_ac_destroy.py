from .termination_condition_base import BaseTerminationCondition


class SAM_AC_destory(BaseTerminationCondition):
    """
    SafeReturn.
    End up the simulation if:
        - the current aircraft has been shot down.
        - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.

        End up the simulation if:
            - the current aircraft has been shot down.
            - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        sam_id = [id for id in env.sams.keys()][0]

        # the current aircraft has crashed
        if env.agents[agent_id].is_shotdown:
            self.log(f'{agent_id} has been shot down! Total Steps={env.current_step}')
            return True, False, info

        elif env.agents[agent_id].is_crash:
            self.log(f'{agent_id} has crashed! Total Steps={env.current_step}')
            return True, False, info

        elif env.sams[sam_id].is_shotdown:
            self.log(f'{sam_id} has been shot down! Total Steps={env.current_step}')
            return True, False, info
        
        elif env.sams[sam_id].is_crash:
            self.log(f'{sam_id} has crashed! Total Steps={env.current_step}')
            return True, False, info

        else:
            return False, False, info
