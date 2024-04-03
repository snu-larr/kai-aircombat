import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        """PPO 알고리즘을 위한 액터 네트워크를 초기화한다.

        이 클래스는 관찰을 기반으로 행동을 결정하는 액터 네트워크를 구현한다. 초기화 과정에서는
        특징 추출 모듈, 선택적 순환 신경망 모듈, 그리고 행동 결정 모듈을 구성한다. 각 모듈의 설정은
        입력 매개변수를 통해 제공된다.

        매개변수:
            args (Namespace): 네트워크 설정을 포함하는 객체. 여기에는 은닉층 크기, 활성화 함수 ID,
                            특징 정규화 사용 여부, 순환 정책 사용 여부 등이 포함될 수 있다.
            obs_space (Space): 관찰 공간의 정의. 네트워크 입력의 형태를 결정한다.
            act_space (Space): 행동 공간의 정의. 네트워크 출력의 형태를 결정한다.
            device (torch.device, optional): 계산에 사용할 디바이스(cpu 또는 cuda). 기본값은 'cpu'.

        속성:
            base (MLPBase): 관찰로부터 특징을 추출하는 모듈.
            rnn (GRULayer, optional): 선택적 순환 신경망 모듈. 순환 정책 사용 시 활성화됨.
            act (ACTLayer): 행동을 결정하는 모듈.

        예시:
            >>> args = Namespace(hidden_size=256, activation_id=2, ...)
            >>> actor = PPOActor(args, obs_space, act_space)
            이 코드는 주어진 설정으로 PPOActor 인스턴스를 생성한다.
        """

        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = args.use_prior
        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):

        """주어진 관찰, 순환 상태, 및 마스크를 기반으로 행동과 로그 확률을 결정한다.

        이 메소드는 네트워크의 순전파 과정을 구현한다. 관찰 데이터와 선택적인 사전 지식을 기반으로
        행동을 결정하며, 순환 신경망 정책을 사용하는 경우, 순환 상태도 함께 업데이트된다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states (Tensor): 순환 신경망(RNN)의 현재 상태.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.
            deterministic (bool, optional): True일 경우 결정적인 행동을 선택, False일 경우 확률적인 행동을 선택. 기본값은 False.

        반환:
            actions (Tensor): 결정된 행동.
            action_log_probs (Tensor): 결정된 행동의 로그 확률.
            rnn_states (Tensor): 업데이트된 순환 신경망의 상태.

        예시:
            >>> obs, rnn_states, masks = ...
            >>> actions, action_log_probs, new_rnn_states = actor.forward(obs, rnn_states, masks, deterministic=True)
            이 코드는 결정적 방식으로 주어진 관찰에 대한 행동과 로그 확률을 결정하고, 필요한 RNN 상태를 업데이트한다.
        """

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11]) # unit degree
            distance = obs[:, 13] * 10000 # unit m
            alpha0 = torch.full(size=(obs.shape[0],1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0],1), fill_value=10).to(**self.tpdv)
            alpha0[distance<=12000] = 6
            alpha0[distance<=8000] = 10
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_prior:
            actions, action_log_probs = self.act(actor_features, deterministic, alpha0=alpha0, beta0=beta0)
        else:
            actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):

        """주어진 행동에 대한 로그 확률과 분포의 엔트로피를 계산한다.

        이 메소드는 관찰 데이터, 순환 상태, 선택된 행동을 기반으로 행동의 로그 확률과
        분포의 엔트로피를 계산한다. 사전 지식을 사용하는 경우, 해당 지식은 행동 결정 과정에
        추가 정보를 제공한다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states (Tensor): 순환 신경망(RNN)의 현재 상태.
            action (Tensor): 평가하려는 행동.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.
            active_masks (Tensor, optional): 특정 시점에서 활성화된 행동만을 평가하기 위한 마스크. None일 경우 모든 행동을 평가.

        반환:
            action_log_probs (Tensor): 주어진 행동의 로그 확률.
            dist_entropy (Tensor): 행동 분포의 엔트로피.

        예시:
            >>> obs, rnn_states, action, masks = ...
            >>> action_log_probs, dist_entropy = actor.evaluate_actions(obs, rnn_states, action, masks)
            이 코드는 주어진 행동에 대한 로그 확률과 분포의 엔트로피를 계산한다.
        """
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11]) # unit degree
            distance = obs[:, 13] * 10000 # unit m
            alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)
            alpha0[distance<=12000] = 6
            alpha0[distance<=8000] = 10
            beta0[attack_angle<=45] = 6
            beta0[attack_angle<=22.5] = 3

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_prior:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks, alpha0=alpha0, beta0=beta0)
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy
