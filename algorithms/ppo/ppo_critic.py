import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.utils import check


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):

        """PPO 알고리즘을 위한 크리틱 네트워크를 초기화한다.

        이 클래스는 주어진 상태의 가치를 평가하는 크리틱 네트워크를 구현한다. 초기화 과정에서는
        특징 추출 모듈, 선택적 순환 신경망 모듈, 그리고 가치 출력 모듈을 구성한다. 각 모듈의 설정은
        입력 매개변수를 통해 제공된다.

        매개변수:
            args (Namespace): 네트워크 설정을 포함하는 객체. 여기에는 은닉층 크기, 활성화 함수 ID,
                            특징 정규화 사용 여부, 순환 정책 사용 여부 등이 포함될 수 있다.
            obs_space (Space): 관찰 공간의 정의. 네트워크 입력의 형태를 결정한다.
            device (torch.device, optional): 계산에 사용할 디바이스(cpu 또는 cuda). 기본값은 'cpu'.

        속성:
            base (MLPBase): 관찰로부터 특징을 추출하는 모듈.
            rnn (GRULayer, optional): 선택적 순환 신경망 모듈. 순환 정책 사용 시 활성화됨.
            value_out (nn.Linear): 상태의 가치를 예측하는 최종 출력 모듈.

        예시:
            >>> args = Namespace(hidden_size=256, activation_id=2, ...)
            >>> critic = PPOCritic(args, obs_space)
            이 코드는 주어진 설정으로 PPOCritic 인스턴스를 생성한다.
        """

        super(PPOCritic, self).__init__()
        # network config
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
        # (3) value module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states, masks):

        """주어진 관찰, 순환 상태, 및 마스크를 기반으로 상태의 가치를 평가한다.

        이 메소드는 네트워크의 순전파 과정을 구현한다. 관찰 데이터를 통해 상태의 가치를 평가하며,
        순환 신경망 정책을 사용하는 경우, 순환 상태도 함께 업데이트된다. 이 과정에서는 특징 추출 모듈,
        선택적 순환 모듈, 그리고 최종적인 가치 예측 모듈을 순차적으로 통과한다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states (Tensor): 순환 신경망(RNN)의 현재 상태.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.

        반환:
            values (Tensor): 계산된 상태의 가치.
            rnn_states (Tensor): 업데이트된 순환 신경망의 상태.

        예시:
            >>> obs, rnn_states, masks = ...
            >>> values, new_rnn_states = critic.forward(obs, rnn_states, masks)
            이 코드는 주어진 관찰, 초기 RNN 상태, 마스크를 기반으로 상태의 가치를 평가하고, 필요한 RNN 상태를 업데이트한다.
        """
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states
