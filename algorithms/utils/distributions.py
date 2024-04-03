import torch
import torch.nn as nn

from .utils import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# Standardize distribution interfaces


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):

        """샘플링 결과에 추가적인 차원을 포함하여 반환한다.

        기본 Categorical 분포의 `sample` 메소드를 오버라이드하여, 샘플링된 텐서에 마지막 차원을 추가한다.

        반환값:
            torch.Tensor: 샘플링된 값들을 포함하는 텐서에 추가적인 차원이 포함된 형태.
        """

        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):

        """주어진 행동에 대한 로그 확률을 계산한다.

        이 메소드는 입력된 행동의 로그 확률을 계산하고, 결과 텐서의 형태를 조정하여 반환한다.
        입력 텐서가 단일 행동이든 배치 형태의 행동이든, 올바른 차원의 로그 확률을 유지할 수 있도록 한다.

        매개변수:
            actions (torch.Tensor): 로그 확률을 계산할 행동의 텐서.

        반환값:
            torch.Tensor: 입력된 행동에 대한 로그 확률이 포함된 텐서.

        예시:
            >>> actions = torch.tensor([[0], [2], [1]])
            >>> distribution = FixedCategorical(logits=torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.25, 0.25, 0.5]]))
            >>> log_probs = distribution.log_probs(actions)
            이 코드는 주어진 행동에 대한 로그 확률을 계산한다.
        """

        # 행동 텐서를 압축하여 불필요한 마지막 차원을 제거한 후, log_prob 계산
        # 계산된 로그 확률의 차원을 조정하고, 마지막 차원을 기준으로 합산하여 반환
        # Single: [1] => [] => [] => [1, 1] => [1] => [1]
        # Batch: [N]/[N, 1] => [N] => [N] => [N, 1] => [N] => [N, 1]

        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.squeeze(-1).unsqueeze(-1).size())
            .sum(-1, keepdim=True)
        )

    def mode(self):

        """확률 분포의 모드(가장 높은 확률을 가지는 행동)를 반환한다.

        이 메소드는 확률 분포에서 가장 확률이 높은 행동의 인덱스를 찾아 반환한다.
        반환된 인덱스는 분포의 `probs` 속성에 따라 결정된다.

        반환값:
            torch.Tensor: 가장 확률이 높은 행동의 인덱스를 나타내는 텐서. 반환되는 텐서는 마지막 차원을 유지한다.

        예시:
            >>> logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.25, 0.25, 0.5]])
            >>> distribution = FixedCategorical(probs=torch.softmax(logits, dim=-1))
            >>> mode_action = distribution.mode()
            이 코드는 주어진 확률 분포에서 각 배치 항목에 대해 가장 확률이 높은 행동의 인덱스를 찾는다.
        """

        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):

        """확률 분포의 엔트로피를 계산하고 반환한다.

        이 메소드는 확률 분포의 불확실성을 나타내는 엔트로피를 계산한다.
        계산된 엔트로피 값에 마지막 차원을 추가하여 반환한다. 이는 일관된 텐서 형태를 유지하기 위함이다.

        반환값:
            torch.Tensor: 계산된 엔트로피 값이 포함된 텐서. 마지막 차원이 추가되어 반환된다.

        예시:
            >>> logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.25, 0.25, 0.5]])
            >>> distribution = FixedCategorical(probs=torch.softmax(logits, dim=-1))
            >>> entropy = distribution.entropy()
            이 코드는 주어진 확률 분포의 엔트로피를 계산한다.
        """

        return super().entropy().unsqueeze(-1)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):

        """주어진 행동의 로그 확률을 계산한다.

        매개변수:
            actions (torch.Tensor): 계산할 행동의 텐서.

        반환값:
            torch.Tensor: 주어진 행동의 로그 확률을 나타내는 텐서. 마지막 차원이 유지된다.
        """

        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):

        """분포의 엔트로피를 계산한다.

        반환값:
            torch.Tensor: 분포의 엔트로피를 나타내는 텐서. 마지막 차원이 유지된다.
        """

        return super().entropy().sum(-1, keepdim=True)

    def mode(self):

        """분포의 모드(평균)를 반환한다.

        반환값:
            torch.Tensor: 분포의 평균을 나타내는 텐서.
        """

        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):

        """주어진 행동의 로그 확률을 계산한다.

        매개변수:
            actions (torch.Tensor): 계산할 행동의 텐서.

        반환값:
            torch.Tensor: 주어진 행동의 로그 확률을 나타내는 텐서. 마지막 차원이 유지된다.
        """

        # Single: [K] => [K] => [1]
        # Batch: [N, K] => [N, K] => [N, 1]
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):

        """분포의 엔트로피를 계산한다.

        반환값:
            torch.Tensor: 분포의 엔트로피를 나타내는 텐서. 마지막 차원이 유지된다.
        """

        return super().entropy().sum(-1, keepdim=True)

    def mode(self):

        """분포의 모드를 반환한다.

        확률이 0.5보다 크면 1을, 그렇지 않으면 0을 반환한다.

        반환값:
            torch.Tensor: 분포의 모드를 나타내는 텐서.
        """

        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):

        """Categorical 클래스의 생성자.

        입력 특성의 차원과 출력 범주의 수를 기반으로 선형 변환 레이어를 초기화한다.
        이 클래스는 입력으로부터 범주형 분포의 로짓을 생성하는데 사용된다.

        매개변수:
            num_inputs (int): 입력 특성의 차원 수.
            num_outputs (int): 출력 범주의 수.
            gain (float): 가중치 초기화에 사용되는 이득 값.

        속성:
            logits_net (nn.Linear): 입력에서 로짓을 생성하기 위한 선형 변환 레이어.
        """

        super(Categorical, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):

        """모델의 순전파를 정의한다.

        입력 텐서 `x`에 대해 선형 변환을 수행하여 로짓을 계산하고, 이를 사용하여 `FixedCategorical` 분포를 생성한다.

        매개변수:
            x (torch.Tensor): 모델의 입력. 형태는 [batch_size, num_inputs]이다.

        반환값:
            FixedCategorical: 입력에 대응하는 로짓을 사용하여 생성된 `FixedCategorical` 분포 객체.

        예시:
            >>> model = Categorical(num_inputs=10, num_outputs=3)
            >>> x = torch.rand(5, 10)  # 배치 크기 5, 입력 차원 10
            >>> distribution = model(x)
            >>> print(distribution)
            FixedCategorical(logits=tensor([...]))
        이 코드는 입력 `x`에 대해 모델을 순전파하고, 결과로 얻은 로짓을 기반으로 `FixedCategorical` 분포를 생성한다.
        """

        x = self.logits_net(x)
        return FixedCategorical(logits=x)

    @property
    def output_size(self) -> int:

        """모델의 출력 크기를 반환한다.

        `Categorical` 모델은 선택된 범주의 인덱스를 나타내는 단일 값으로, 출력 크기는 항상 1이다.

        반환값:
            int: 모델의 출력 크기, 항상 1.

        예시:
            >>> model = Categorical(num_inputs=10, num_outputs=3)
            >>> print(model.output_size)
            1
        이 예시에서 `Categorical` 모델의 `output_size` 프로퍼티는 항상 1을 반환한다.
        """

        return 1


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):

        """DiagGaussian 클래스의 생성자.

        주어진 입력 차원에 대해 대각 가우시안 분포의 파라미터(평균과 로그 표준편차)를 계산하는 모델을 초기화한다.

        매개변수:
            num_inputs (int): 입력 차원의 수.
            num_outputs (int): 출력 차원의 수, 즉 가우시안 분포의 차원.
            gain (float): 가중치 초기화에 사용될 이득(gain) 값.

        속성:
            mu_net (nn.Linear): 입력에서 분포의 평균을 계산하는 선형 레이어.
            log_std (nn.Parameter): 분포의 로그 표준편차를 나타내는 학습 가능한 파라미터.
            _num_outputs (int): 출력 차원의 수, 내부적으로 사용됨.
        """

        super(DiagGaussian, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.mu_net = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):

        """모델의 순전파를 정의하고, 입력에 대한 정규 분포 파라미터를 반환한다.

        입력 텐서 `x`를 받아, 모델은 각 행동의 평균을 계산하고, 학습 가능한 파라미터로부터 표준편차를 계산한다.
        이 정보는 연속적인 행동 공간에서의 행동 선택을 위해 정규 분포 객체를 생성하는 데 사용된다.

        매개변수:
            x (torch.Tensor): 모델의 입력.

        반환값:
            FixedNormal: 입력에 대응하는 정규 분포 파라미터(평균과 표준편차)를 가진 `FixedNormal` 객체.

        예시:
            >>> model = DiagGaussian(num_inputs=10, num_outputs=2)
            >>> x = torch.rand(5, 10)  # 배치 크기 5, 입력 차원 10
            >>> distribution = model(x)
            >>> print(distribution)
            FixedNormal(...)
        이 코드는 입력 `x`에 대해 모델을 순전파하고, 결과로 얻은 평균과 표준편차를 가진 `FixedNormal` 분포 객체를 생성한다.
        """

        action_mean = self.mu_net(x)
        return FixedNormal(action_mean, self.log_std.exp())

    @property
    def output_size(self) -> int:

        """모델의 출력 크기를 반환한다.

        이 값은 정규 분포의 차원을 나타내며, 강화 학습에서 에이전트가 선택할 수 있는 연속적인 행동 공간의 크기를 의미한다.

        반환값:
            int: 모델의 출력 차원 수.

        예시:
            >>> model = DiagGaussian(num_inputs=10, num_outputs=2)
            >>> print(model.output_size)
            2
        이 예시에서는 입력 차원이 10이고 출력(행동 공간) 차원이 2인 `DiagGaussian` 모델의 출력 크기를 조회한다.
        """
        
        return self._num_outputs

class BetaShootBernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        """BetaShootBernoulli 클래스의 생성자.

        입력 특성에 기반하여 베타-슈팅-베르누이 분포의 파라미터를 계산하는 모델을 초기화한다.

        매개변수:
            num_inputs (int): 입력 특성의 차원 수.
            num_outputs (int): 출력 파라미터의 차원 수. 이는 베타 분포의 파라미터 수와 베르누이 분포의 결정을 포함한다.
            gain (float): 가중치 초기화에 사용될 이득 값.

        속성:
            net (nn.Linear): 입력에서 분포의 파라미터를 계산하는 선형 변환 레이어.
            _num_outputs (int): 출력 파라미터의 차원 수를 내부적으로 저장.
            constraint (nn.Softplus): 분포의 파라미터에 제약을 가하는 함수.
        """
        super(BetaShootBernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs
        self.constraint = nn.Softplus()

    def forward(self, x, **kwargs):
        
        """모델의 순전파를 정의하고, 베타-슈팅-베르누이 분포의 파라미터를 기반으로 한 확률을 반환한다.

        입력 텐서 `x`를 처리하여, 베타 분포의 파라미터 `alpha`와 `beta`를 계산하고,
        이를 바탕으로 베르누이 분포의 확률 `p`를 계산한다. 계산된 확률은 슈팅 결정을 모델링하는 데 사용된다.

        매개변수:
            x (torch.Tensor): 모델의 입력.
            **kwargs: `alpha0`과 `beta0`를 포함하는 추가 파라미터. 이들은 베타 분포의 파라미터에 더해져
                    최종 확률 계산에 사용된다.

        반환값:
            FixedBernoulli: 계산된 확률 `p`를 사용하여 생성된 `FixedBernoulli` 객체.

        예시:
            >>> model = BetaShootBernoulli(num_inputs=10, num_outputs=2)
            >>> x = torch.rand(5, 10)  # 배치 크기 5, 입력 차원 10
            >>> distribution = model(x, alpha0=1.0, beta0=1.0)
            >>> print(distribution)
            FixedBernoulli(...)
        이 코드는 입력 `x`와 추가 파라미터 `alpha0`, `beta0`를 사용하여 모델을 순전파하고,
        결과로 얻은 확률을 가진 `FixedBernoulli` 분포 객체를 생성한다.
        """
        
        x = self.net(x)
        x = self.constraint(x) # contrain alpha, beta >=0
        x = 100 - self.constraint(100-x) # constrain alpha, beta <=100
        alpha = 1 + x[:, 0].unsqueeze(-1)
        beta = 1 + x[:, 1].unsqueeze(-1)
        alpha_0 = kwargs['alpha0']
        beta_0 = kwargs['beta0']
        # print(f"{alpha}, {beta}, {alpha_0}, {beta_0}")
        p = (alpha + alpha_0) / (alpha + alpha_0 + beta + beta_0)
        return FixedBernoulli(p)

    @property
    def output_size(self) -> int:

        """모델의 출력 크기를 반환한다.

    이 프로퍼티는 모델이 생성하는 분포의 차원 수, 즉 가능한 행동의 수나 베르누이 결정의 수를 나타낸다.
    `BetaShootBernoulli` 클래스의 경우, 이 값은 베타 분포의 모수를 계산하는 데 사용되는 출력 차원 수를 의미한다.

    반환값:
        int: 모델이 생성하는 분포의 차원 수를 나타내는 정수.

    예시:
        >>> model = BetaShootBernoulli(num_inputs=10, num_outputs=2)
        >>> print(model.output_size)
        2
    이 예시에서는 `BetaShootBernoulli` 모델의 출력 크기가 2임을 보여준다. 이는 모델이 두 개의 파라미터(예를 들어, 알파와 베타)를 사용하여 베타 분포를 생성함을 의미한다.
    """
        
        return self._num_outputs

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):

        """Bernoulli 클래스의 생성자.

        주어진 입력 차원에 대해 베르누이 분포의 로짓을 생성하는 선형 레이어를 초기화한다.

        매개변수:
            num_inputs (int): 모델 입력 차원의 수.
            num_outputs (int): 베르누이 분포의 출력 차원의 수. 이는 모델링하려는 이진 결정의 수와 같다.
            gain (float): 가중치 초기화에 사용될 이득 값.

        속성:
            logits_net (nn.Linear): 입력에서 베르누이 분포의 로짓을 생성하는 선형 변환 레이어.
            _num_outputs (int): 출력 차원의 수를 내부적으로 저장하는 변수.
        """
        
        super(Bernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):

        """모델의 순전파를 정의하고, 입력에 대한 베르누이 분포 객체를 반환한다.

        입력 텐서 `x`에 대해 선형 변환을 수행하여 로짓을 계산하고, 이를 사용하여 `FixedBernoulli` 분포를 생성한다.

        매개변수:
            x (torch.Tensor): 모델의 입력. 형태는 [batch_size, num_inputs]이다.

        반환값:
            FixedBernoulli: 계산된 로짓을 사용하여 생성된 `FixedBernoulli` 분포 객체.

        예시:
            >>> model = Bernoulli(num_inputs=10, num_outputs=3)
            >>> x = torch.rand(5, 10)  # 배치 크기 5, 입력 차원 10
            >>> distribution = model(x)
            >>> print(distribution)
            FixedBernoulli(logits=tensor([...]))
        이 코드는 입력 `x`에 대해 모델을 순전파하고, 결과로 얻은 로짓을 기반으로 `FixedBernoulli` 분포를 생성한다.
        """

        x = self.logits_net(x)
        return FixedBernoulli(logits=x)

    @property
    def output_size(self) -> int:

        """모델이 생성할 수 있는 이진 결정의 수를 반환한다.

        이 값은 모델의 출력 차원을 나타내며, `Bernoulli` 모델에서는 이진 결정(0 또는 1)의 수와 동일하다.

        반환값:
            int: 모델이 생성할 수 있는 이진 결정의 수.

        예시:
            >>> model = Bernoulli(num_inputs=10, num_outputs=3)
            >>> print(model.output_size)
            3
        이 예시에서 `Bernoulli` 모델은 3개의 이진 결정을 생성할 수 있다.
        """
        
        return self._num_outputs
