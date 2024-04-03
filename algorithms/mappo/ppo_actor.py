import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        """PPO 알고리즘용 액터 네트워크를 초기화한다.

        이 생성자는 PPO 액터 네트워크의 구성을 정의한다. 이 네트워크는 특징 추출, 순환 신경망(RNN), 
        행동 결정의 세 가지 주요 모듈로 구성된다. 각 모듈은 환경으로부터의 관찰을 입력으로 받아
        액션을 결정하는데 필요한 처리를 수행한다.

        매개변수:
            args (Namespace): 알고리즘 구성 및 네트워크 설정을 포함하는 객체.
            - gain (float): 네트워크의 가중치 초기화에 사용되는 이득(gain) 값.
            - hidden_size (int): 특징 추출 모듈의 은닉층 크기.
            - act_hidden_size (int): 행동 결정 모듈의 은닉층 크기.
            - activation_id (int): 사용할 활성화 함수의 ID.
            - use_feature_normalization (bool): 특징 정규화 사용 여부.
            - use_recurrent_policy (bool): 순환 정책 사용 여부.
            - recurrent_hidden_size (int): 순환 모듈의 은닉 상태 크기.
            - recurrent_hidden_layers (int): 순환 모듈의 은닉층 개수.

            obs_space (Space): 관찰 공간의 정의. 환경으로부터의 입력 형태를 정의한다.
            act_space (Space): 행동 공간의 정의. 가능한 행동의 형태를 정의한다.
            device (torch.device, optional): 계산에 사용할 디바이스(cpu 또는 cuda). 기본값은 'cpu'.

        주요 속성:
            - base (MLPBase): 관찰 데이터로부터 특징을 추출하는 모듈.
            - rnn (GRULayer): 선택적 순환 신경망 모듈. use_recurrent_policy가 True일 때만 사용.
            - act (ACTLayer): 최종적으로 행동을 결정하는 모듈.

        예시:
            >>> args = Namespace(gain=0.01, hidden_size=256, act_hidden_size=128, activation_id=1,
                                use_feature_normalization=True, use_recurrent_policy=False,
                                recurrent_hidden_size=64, recurrent_hidden_layers=1)
            >>> actor = PPOActor(args, obs_space, act_space)
            이 코드는 주어진 설정으로 PPOActor 네트워크의 인스턴스를 생성한다.
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
        
        """네트워크를 통해 주어진 관찰로부터 행동을 생성한다.

        이 메소드는 네트워크의 순전파 과정을 구현한다. 관찰(obs), 순환 네트워크 상태(rnn_states),
        마스크(masks)를 입력으로 받아, 결정한 행동, 행동의 로그 확률, 그리고 순환 네트워크 상태를 반환한다.
        순환 정책을 사용하는 경우, rnn_states와 masks가 순환 네트워크의 상태 및 시퀀스 간의 연속성을 유지하는데 사용된다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states (Tensor): 순환 신경망(RNN)의 현재 상태. 순환 정책을 사용하지 않는 경우 무시될 수 있음.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크. 순환 정책을 사용하는 경우 필요.
            deterministic (bool, optional): True일 경우 결정적인 행동을 선택, False일 경우 확률적인 행동을 선택. 기본값은 False.

        반환:
            tuple: 
            - actions (Tensor): 결정된 행동.
            - action_log_probs (Tensor): 행동의 로그 확률.
            - rnn_states (Tensor): 업데이트된 순환 신경망의 상태.

        예시:
            >>> obs = torch.randn(1, obs_space.shape[0])
            >>> rnn_states = torch.zeros(1, self.recurrent_hidden_size)
            >>> masks = torch.ones(1, 1)
            >>> actions, log_probs, new_rnn_states = actor.forward(obs, rnn_states, masks)
            이 예시는 주어진 관찰(obs), 초기 RNN 상태(rnn_states), 마스크(masks)로부터 
            행동과 로그 확률, 업데이트된 RNN 상태를 반환하는 과정을 보여준다.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy
