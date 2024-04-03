import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.utils import check


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        """PPO 크리틱 네트워크를 초기화한다.

        이 생성자는 PPO 크리틱 네트워크의 구성을 정의한다. 네트워크는 특징 추출 모듈, 
        선택적 순환 신경망(RNN) 모듈, 그리고 가치(value) 출력 모듈로 구성된다. 이 구조는 
        주어진 상태에서 에이전트가 받을 것으로 예상되는 미래 보상의 총합을 추정하는데 사용된다.

        매개변수:
            args (Namespace): 네트워크 설정을 포함하는 객체.
            - hidden_size (int): 특징 추출 모듈의 은닉층 크기.
            - act_hidden_size (int): 가치 예측 모듈의 은닉층 크기.
            - activation_id (int): 활성화 함수의 ID.
            - use_feature_normalization (bool): 특징 정규화 사용 여부.
            - use_recurrent_policy (bool): 순환 정책 사용 여부.
            - recurrent_hidden_size (int): 순환 모듈의 은닉 상태 크기.
            - recurrent_hidden_layers (int): 순환 모듈의 은닉층 개수.
            obs_space (Space): 관찰 공간의 정의. 환경으로부터의 입력 형태를 정의한다.
            device (torch.device, optional): 계산에 사용할 디바이스(cpu 또는 cuda). 기본값은 'cpu'.

        주요 속성:
            - base (MLPBase): 관찰 데이터로부터 특징을 추출하는 모듈.
            - rnn (GRULayer): 선택적 순환 신경망 모듈. use_recurrent_policy가 True일 때만 사용.
            - value_out (nn.Linear): 상태의 가치를 예측하는 최종 출력 모듈.

        예시:
            >>> args = Namespace(hidden_size=256, act_hidden_size=[128, 64], activation_id=1,
                                use_feature_normalization=True, use_recurrent_policy=True,
                                recurrent_hidden_size=64, recurrent_hidden_layers=1)
            >>> critic = PPOCritic(args, obs_space)
            이 코드는 주어진 설정으로 PPOCritic 네트워크의 인스턴스를 생성한다.
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
