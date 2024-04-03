import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        """PPOPolicy 클래스의 인스턴스를 초기화한다.

        이 생성자는 PPO 알고리즘에 사용되는 액터와 크리틱 네트워크를 초기화하고,
        두 네트워크의 매개변수를 최적화하기 위한 옵티마이저를 설정한다.

        매개변수:
            args (Namespace): 학습 설정을 포함하는 객체. 여기에는 학습률 등의 매개변수가 포함될 수 있다.
            obs_space (Space): 관찰 공간의 정의.
            act_space (Space): 행동 공간의 정의.
            device (torch.device, optional): 계산에 사용할 디바이스(cpu 또는 cuda). 기본값은 'cpu'.

        속성:
            args: 학습과 관련된 설정을 담고 있는 객체.
            device: 계산에 사용될 장치.
            obs_space: 관찰 공간의 정의.
            act_space: 행동 공간의 정의.
            actor: 관찰을 기반으로 행동을 결정하는 액터 네트워크.
            critic: 주어진 상태의 가치를 평가하는 크리틱 네트워크.
            optimizer: 액터와 크리틱 네트워크의 매개변수를 최적화하기 위한 옵티마이저.

        예시:
            >>> args = Namespace(lr=0.01, hidden_size=256, ...)
            >>> policy = PPOPolicy(args, obs_space, act_space)
            이 코드는 주어진 설정으로 PPOPolicy 인스턴스를 생성한다.
        """

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        
        """주어진 관찰과 순환 상태를 기반으로 행동, 가치, 및 로그 확률을 결정한다.

        이 메소드는 액터 네트워크를 통해 관찰을 기반으로 행동과 해당 행동의 로그 확률을 결정하고,
        크리틱 네트워크를 통해 주어진 상태의 가치를 평가한다. 순환 신경망 정책을 사용하는 경우,
        순환 상태도 함께 업데이트되며, 이는 시퀀스 간의 정보를 유지하는 데 도움이 된다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states_actor (Tensor): 액터 네트워크의 순환 신경망 상태.
            rnn_states_critic (Tensor): 크리틱 네트워크의 순환 신경망 상태.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.

        반환:
            values (Tensor): 계산된 상태의 가치.
            actions (Tensor): 결정된 행동.
            action_log_probs (Tensor): 결정된 행동의 로그 확률.
            rnn_states_actor (Tensor): 업데이트된 액터 네트워크의 순환 신경망 상태.
            rnn_states_critic (Tensor): 업데이트된 크리틱 네트워크의 순환 신경망 상태.

        예시:
            >>> obs, rnn_states_actor, rnn_states_critic, masks = ...
            >>> values, actions, action_log_probs, new_rnn_states_actor, new_rnn_states_critic = policy.get_actions(
                    obs, rnn_states_actor, rnn_states_critic, masks)
            이 코드는 주어진 관찰 및 순환 상태로부터 행동, 가치, 및 로그 확률을 결정하고, 필요한 RNN 상태를 업데이트한다.
        """
        
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        
        """주어진 관찰과 순환 상태를 기반으로 상태의 가치를 평가한다.

        이 메소드는 크리틱 네트워크를 사용하여 주어진 관찰로부터 상태의 가치를 평가한다.
        순환 신경망 정책을 사용하는 경우, 순환 상태 정보를 활용하여 시퀀스 데이터에 대한 가치 추정이 가능하다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states_critic (Tensor): 크리틱 네트워크의 순환 신경망 상태.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.

        반환:
            values (Tensor): 계산된 상태의 가치.

        예시:
            >>> obs, rnn_states_critic, masks = ...
            >>> values = policy.get_values(obs, rnn_states_critic, masks)
            이 코드는 주어진 관찰 및 순환 상태로부터 상태의 가치를 평가한다.
        """

        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):

        """주어진 행동에 대한 가치, 로그 확률, 및 분포의 엔트로피를 평가한다.

        이 메소드는 액터 네트워크를 통해 주어진 행동의 로그 확률과 분포의 엔트로피를 계산하고,
        크리틱 네트워크를 통해 해당 행동의 상태 가치를 평가한다. 순환 신경망 정책을 사용하는 경우,
        순환 상태 정보도 함께 처리된다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states_actor (Tensor): 액터 네트워크의 순환 신경망 상태.
            rnn_states_critic (Tensor): 크리틱 네트워크의 순환 신경망 상태.
            action (Tensor): 평가하려는 행동.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.
            active_masks (Tensor, optional): 특정 시점에서 활성화된 행동만을 평가하기 위한 마스크. None일 경우 모든 행동을 평가.

        반환:
            values (Tensor): 계산된 상태의 가치.
            action_log_probs (Tensor): 주어진 행동의 로그 확률.
            dist_entropy (Tensor): 행동 분포의 엔트로피.

        예시:
            >>> obs, rnn_states_actor, rnn_states_critic, action, masks = ...
            >>> values, action_log_probs, dist_entropy = policy.evaluate_actions(
                    obs, rnn_states_actor, rnn_states_critic, action, masks)
            이 코드는 주어진 행동에 대한 가치, 로그 확률, 및 엔트로피를 평가한다.
        """

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        
        """주어진 관찰과 순환 상태를 기반으로 행동을 결정한다.

        이 메소드는 액터 네트워크를 사용하여 주어진 관찰로부터 행동을 결정하고,
        순환 신경망 정책을 사용하는 경우 순환 상태도 함께 업데이트한다. 결정적 방식을 선택할 수 있으며,
        이는 환경과의 상호작용 시 더 일관된 행동을 취하고자 할 때 유용하다.

        매개변수:
            obs (Tensor): 네트워크에 입력되는 관찰 값. 환경으로부터의 관찰 데이터.
            rnn_states_actor (Tensor): 액터 네트워크의 순환 신경망 상태.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.
            deterministic (bool, optional): True일 경우 결정적인 행동을 선택, False일 경우 확률적인 행동을 선택. 기본값은 False.

        반환:
            actions (Tensor): 결정된 행동.
            rnn_states_actor (Tensor): 업데이트된 액터 네트워크의 순환 신경망 상태.

        예시:
            >>> obs, rnn_states_actor, masks = ...
            >>> actions, new_rnn_states_actor = policy.act(obs, rnn_states_actor, masks, deterministic=True)
            이 코드는 결정적 방식으로 주어진 관찰에 대한 행동을 결정하고, 필요한 RNN 상태를 업데이트한다.
        """

        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):

        """액터와 크리틱 네트워크를 학습 모드로 설정한다.

        이 메소드는 훈련 세션을 시작하기 전에 호출되어야 하며, 액터와 크리틱 네트워크를 학습 모드로 전환한다.
        학습 모드에서는 드롭아웃과 같은 특정 기능들이 활성화될 수 있다.

        반환값: 없음.
        """

        self.actor.train()
        self.critic.train()

    def prep_rollout(self):

        """액터와 크리틱 네트워크를 평가 모드로 설정한다.

        이 메소드는 평가 세션 또는 실제 환경에서의 롤아웃을 시작하기 전에 호출되어야 하며,
        액터와 크리틱 네트워크를 평가 모드로 전환한다. 평가 모드에서는 드롭아웃과 같은 특정 기능들이 비활성화된다.

        반환값: 없음.
        """

        self.actor.eval()
        self.critic.eval()

    def copy(self):

        """PPOPolicy 인스턴스의 복사본을 생성하여 반환한다.

        이 메소드는 현재 PPOPolicy 인스턴스의 상태를 그대로 복제한 새 인스턴스를 생성하고 반환한다.
        이는 주로 훈련 과정에서 여러 정책을 동시에 실험하고자 할 때 유용하게 사용될 수 있다.

        반환값:
            PPOPolicy: 현재 인스턴스의 복사본.

        예시:
            >>> new_policy = current_policy.copy()
            이 코드는 현재 PPOPolicy 인스턴스의 복사본을 생성한다.
        """

        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
