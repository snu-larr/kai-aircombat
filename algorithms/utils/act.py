import torch
import torch.nn as nn
import gymnasium as gym
from .mlp import MLPLayer
from .distributions import BetaShootBernoulli, Categorical, DiagGaussian, Bernoulli


class ACTLayer(nn.Module):
    def __init__(self, act_space, input_dim, hidden_size, activation_id, gain):

        """ACTLayer 클래스의 인스턴스를 초기화한다.

        다양한 유형의 행동 공간을 지원하는 액션 레이어를 구성한다. 이 클래스는 입력 차원, 숨겨진 차원,
        활성화 함수 ID, 그리고 이득을 바탕으로 주어진 행동 공간에 맞는 출력 레이어를 설정한다.

        매개변수:
            act_space (gym.spaces): Gym 행동 공간 객체. 지원되는 공간 유형에는 Discrete, Box, MultiBinary,
                                    MultiDiscrete, Tuple 등이 있다.
            input_dim (int): 네트워크 입력 차원.
            hidden_size (list of int): 숨겨진 레이어의 차원. 빈 리스트일 경우 MLP 레이어는 사용되지 않는다.
            activation_id (int): 활성화 함수의 ID. 특정 활성화 함수를 선택하는 데 사용된다.
            gain (float): 레이어 초기화에 사용되는 이득 값.

        속성:
            mlp (MLPLayer, optional): 숨겨진 레이어를 포함하는 경우의 MLP 레이어.
            action_out (nn.Module, optional): 행동 공간이 Discrete, Box, 또는 MultiBinary인 경우의 출력 레이어.
            action_outs (nn.ModuleList, optional): 행동 공간이 MultiDiscrete 또는 Tuple인 경우의 출력 레이어 리스트.
        """

        super(ACTLayer, self).__init__()
        self._mlp_actlayer = False
        self._continuous_action = False
        self._multidiscrete_action = False
        self._mixed_action = False
        self._shoot_action = False
        self._trigger_dim = 0

        if len(hidden_size) > 0:
            self._mlp_actlayer = True
            self.mlp = MLPLayer(input_dim, hidden_size, activation_id)
            input_dim = self.mlp.output_size

        if isinstance(act_space, gym.spaces.Discrete):
            action_dim = act_space.n
            self.action_out = Categorical(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.Box):
            self._continuous_action = True
            action_dim = act_space.shape[0]
            self.action_out = DiagGaussian(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.MultiBinary):
            action_dim = act_space.shape[0]
            self.action_out = Bernoulli(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            self._multidiscrete_action = True
            action_dims = act_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(Categorical(input_dim, action_dim, gain))
            self.action_outs = nn.ModuleList(action_outs)
        elif isinstance(act_space, gym.spaces.Tuple) and len(act_space.spaces) > 2:
            self._shoot_action = True
            discrete_dims = act_space[0].nvec
            self._discrete_dim = act_space[0].shape[0]
            self._control_shoot_dim = 2
            self._trigger_dim = len(act_space.spaces) - 1 # ONLY ONE MULTIDISCRETE
            action_outs = []
            for discrete_dim in discrete_dims:
                action_outs.append(Categorical(input_dim, discrete_dim, gain))
            
            for _ in range(self._trigger_dim):
                action_outs.append(BetaShootBernoulli(input_dim, self._control_shoot_dim, gain))
            self.action_outs = nn.ModuleList(action_outs)
        
        elif isinstance(act_space, gym.spaces.Tuple) and  \
              isinstance(act_space[0], gym.spaces.MultiDiscrete) and \
                  isinstance(act_space[1], gym.spaces.Discrete):
            # NOTE: only for shoot missile
            self._shoot_action = True
            discrete_dims = act_space[0].nvec
            self._discrete_dim = act_space[0].shape[0]
            self._control_shoot_dim = 2
            self._trigger_dim = 1
            action_outs = []
            for discrete_dim in discrete_dims:
                action_outs.append(Categorical(input_dim, discrete_dim, gain))
            action_outs.append(BetaShootBernoulli(input_dim, self._control_shoot_dim, gain))
            self.action_outs = nn.ModuleList(action_outs)
        else: 
            raise NotImplementedError(f"Unsupported action space type: {type(act_space)}!")

    def forward(self, x, deterministic=False, **kwargs):

        """주어진 입력으로부터 행동과 행동의 로그 확률을 계산한다.

        다양한 행동 공간 유형을 지원하며, 선택적으로 결정론적 행동 선택 기능을 제공한다. 
        복합 행동 공간 또는 특정 요구사항(예: 발사 메커니즘 제어)을 위한 추가 인자는 `kwargs`를 통해 처리된다.

        매개변수:
            x (torch.Tensor): 네트워크에 입력되는 데이터.
            deterministic (bool, optional): True일 경우, 행동 분포의 모드(가장 가능성 높은 행동)를 반환하고,
                                            False일 경우 행동 분포로부터 샘플링한다. 기본값은 False.

        반환:
            actions (torch.Tensor): 결정된 행동.
            action_log_probs (torch.Tensor): 결정된 행동의 로그 확률.

        예시:
            >>> x = torch.randn(size=(1, input_dim))
            >>> actions, action_log_probs = act_layer.forward(x, deterministic=True)
            이 코드는 주어진 입력 `x`에 대해 결정론적으로 행동과 해당 로그 확률을 계산한다.
        """

        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
        
        elif self._shoot_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs[:self._discrete_dim]:
                action_dist = action_out(x)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            for shoot_action_out in self.action_outs[self._discrete_dim:]:
                shoot_action_dist = shoot_action_out(x, **kwargs)
                shoot_action = shoot_action_dist.mode() if deterministic else shoot_action_dist.sample()
                actions.append(shoot_action)
            
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dists = self.action_out(x)
            actions = action_dists.mode() if deterministic else action_dists.sample()
            action_log_probs = action_dists.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, active_masks=None, **kwargs):
        
        """주어진 행동의 로그 확률과 분포의 엔트로피를 계산한다.

        이 메소드는 다양한 행동 공간 유형에 대응하여 주어진 행동의 로그 확률과 해당 행동 분포의 엔트로피를 계산한다.
        선택적으로, 활성화 마스크를 사용하여 특정 에이전트가 활성 상태인 경우에만 행동 평가를 진행할 수 있다.

        매개변수:
            x (torch.Tensor): 네트워크에 입력되는 데이터.
            action (torch.Tensor): 엔트로피와 로그 확률을 평가하고자 하는 행동.
            active_masks (torch.Tensor, optional): 에이전트가 활성 상태인지를 나타내는 마스크. None일 경우 모든 에이전트가 활성 상태로 간주.

        반환:
            action_log_probs (torch.Tensor): 입력된 행동의 로그 확률.
            dist_entropy (torch.Tensor): 주어진 입력에 대한 행동 분포의 엔트로피.

        예시:
            >>> x, action = torch.randn(size=(1, input_dim)), torch.randint(low=0, high=action_dim, size=(1,))
            >>> action_log_probs, dist_entropy = act_layer.evaluate_actions(x, action)
            이 코드는 주어진 행동 `action`에 대한 로그 확률과 분포의 엔트로피를 계산한다.
        """

        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_dist = action_out(x)
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        elif self._shoot_action:
            dis_action, trigger_actions_bundle = action.split((self._discrete_dim, self._trigger_dim), dim=-1)
            trigger_actions = torch.chunk(trigger_actions_bundle, chunks = 8, dim = 1)
            action_log_probs = []
            dist_entropy = []
            # multi-discrete action
            dis_action = torch.transpose(dis_action, 0, 1)
            for action_out, act in zip(self.action_outs[:self._discrete_dim], dis_action):
                action_dist = action_out(x)
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))

            # shoot action
            for trigger_action, shoot_action_out in zip(trigger_actions, self.action_outs[self._discrete_dim:]):
                shoot_action_dist = shoot_action_out(x, **kwargs)
                action_log_probs.append(shoot_action_dist.log_probs(trigger_action))
                if active_masks is not None:
                    dist_entropy.append((shoot_action_dist.entropy() * active_masks) / active_masks.sum())
                else:
                    dist_entropy.append(shoot_action_dist.entropy() / action_log_probs[-1].size(0))

            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks) / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy() / action_log_probs.size(0)
        return action_log_probs, dist_entropy

    def get_probs(self, x):
        
        """입력으로부터 행동 확률을 계산한다.

        다중 이산 행동 공간에 대해 각 행동의 확률을 계산하고 반환한다.
        연속 행동 공간이나 특수한 행동 공간(예: 발사 메커니즘 제어)에는 사용할 수 없다.

        매개변수:
            x (torch.Tensor): 네트워크에 입력되는 데이터.

        반환값:
            action_probs (torch.Tensor): 계산된 행동의 확률.

        예외:
            ValueError: 연속 행동 공간 또는 발사 메커니즘 제어와 같은 특수 행동 공간에서 호출될 경우 발생.
        """

        if self._mlp_actlayer:
            x = self.mlp(x)
        if self._multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action_prob = action_dist.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, dim=-1)
        elif self._continuous_action or self._shoot_action:
            raise ValueError("Normal distribution has no `probs` attribute!")
        else:
            action_dists = self.action_out(x)
            action_probs = action_dists.probs
        return action_probs

    @property
    def output_size(self) -> int:

        """모델 출력의 크기를 반환한다.

        행동 공간의 유형(다중 이산 행동 공간, 발사 메커니즘 제어 등)에 따라
        모델 출력의 크기가 결정된다.

        반환값:
            output_size (int): 모델 출력의 크기.
        """
        
        if self._multidiscrete_action or self._shoot_action:
            return len(self.action_outs)
        else:
            return self.action_out.output_size
