import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):

        """PPOPolicy 클래스의 인스턴스를 초기화한다.

        이 생성자는 PPO 알고리즘을 구현하는데 필요한 구성 요소를 초기화한다. 액터(행동 결정)와 
        크리틱(가치 평가) 네트워크를 설정하고, 이를 학습시키기 위한 Adam 옵티마이저를 구성한다. 
        또한, 관찰 공간, 중앙 관찰 공간(다중 에이전트 설정에서 사용), 행동 공간 및 계산 디바이스를 설정한다.

        매개변수:
            args (Namespace): 학습 및 네트워크 구성을 위한 설정을 포함하는 객체.
            - lr (float): 학습률.

            obs_space (Space): 개별 에이전트의 관찰 공간.
            cent_obs_space (Space): 중앙 관찰 공간, 다중 에이전트 환경에서 여러 에이전트의 관찰을 통합.
            act_space (Space): 가능한 행동의 공간.
            device (torch.device, optional): 계산에 사용할 디바이스(cpu 또는 cuda). 기본값은 'cpu'.

        속성:
            actor (PPOActor): 환경으로부터의 관찰을 기반으로 행동을 결정하는 네트워크.
            critic (PPOCritic): 주어진 상태의 가치를 평가하는 네트워크.
            optimizer (torch.optim.Adam): 액터와 크리틱 네트워크의 매개변수를 최적화하기 위한 옵티마이저.

        예시:
            >>> args = Namespace(lr=0.01, hidden_size=256, ...)
            >>> policy = PPOPolicy(args, obs_space, cent_obs_space, act_space)
            이 코드는 주어진 설정으로 PPOPolicy 인스턴스를 생성한다.
        """

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr
        
        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.cent_obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks):
        
        """중앙 관찰 및 개별 관찰을 기반으로 행동과 가치를 계산한다.

        이 메소드는 액터와 크리틱 네트워크를 사용하여 주어진 관찰로부터 행동, 행동의 로그 확률,
        상태 가치를 계산한다. 순환 정책을 사용하는 경우, 순환 신경망(RNN) 상태도 함께 업데이트되어 반환된다.

        매개변수:
            cent_obs (Tensor): 중앙 관찰 데이터. 여러 에이전트의 정보를 통합한 관찰 데이터.
            obs (Tensor): 개별 에이전트의 관찰 데이터.
            rnn_states_actor (Tensor): 액터 네트워크의 순환 신경망 상태.
            rnn_states_critic (Tensor): 크리틱 네트워크의 순환 신경망 상태.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.

        반환:
            values (Tensor): 주어진 상태의 가치.
            actions (Tensor): 결정된 행동.
            action_log_probs (Tensor): 결정된 행동의 로그 확률.
            rnn_states_actor (Tensor): 업데이트된 액터 네트워크의 순환 신경망 상태.
            rnn_states_critic (Tensor): 업데이트된 크리틱 네트워크의 순환 신경망 상태.

        예시:
            >>> cent_obs, obs, rnn_states_actor, rnn_states_critic, masks = ...
            >>> values, actions, action_log_probs, new_rnn_states_actor, new_rnn_states_critic = policy.get_actions(
                    cent_obs, obs, rnn_states_actor, rnn_states_critic, masks)
            이 코드는 주어진 관찰 및 상태로부터 행동과 가치를 계산하고, 필요한 상태 정보를 업데이트한다.
        """

        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        
        """중앙 관찰 데이터를 기반으로 상태의 가치를 계산한다.

        이 메소드는 크리틱 네트워크를 사용하여 주어진 중앙 관찰 데이터로부터 상태의 가치를 추정한다.
        순환 정책을 사용하는 경우, 주어진 순환 신경망(RNN) 상태와 마스크를 이용하여 시퀀스 간의 정보를 유지한다.

        매개변수:
            cent_obs (Tensor): 중앙 관찰 데이터. 여러 에이전트의 정보를 통합한 관찰 데이터.
            rnn_states_critic (Tensor): 크리틱 네트워크의 순환 신경망 상태.
            masks (Tensor): 시퀀스의 요소 간의 연속성을 나타내는 마스크.

        반환:
            values (Tensor): 계산된 상태의 가치.

        예시:
            >>> cent_obs, rnn_states_critic, masks = ...
            >>> values = policy.get_values(cent_obs, rnn_states_critic, masks)
            이 코드는 주어진 중앙 관찰 데이터로부터 상태의 가치를 계산한다.
        """
        
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        
        """주어진 행동의 가치, 로그 확률, 및 분포의 엔트로피를 평가한다.

        이 메소드는 액터 네트워크를 사용하여 주어진 행동의 로그 확률과 분포의 엔트로피를 계산하며,
        크리틱 네트워크를 사용하여 상태의 가치를 추정한다. 순환 정책을 사용하는 경우, 순환 신경망(RNN) 상태가
        업데이트되며, 선택적으로 특정 행동만을 평가하기 위한 활성화 마스크를 사용할 수 있다.

        매개변수:
            cent_obs (Tensor): 중앙 관찰 데이터. 여러 에이전트의 정보를 통합한 관찰 데이터.
            obs (Tensor): 개별 에이전트의 관찰 데이터.
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
            >>> cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks = ...
            >>> values, action_log_probs, dist_entropy = policy.evaluate_actions(
                    cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks)
            이 코드는 주어진 행동과 관찰 데이터로부터 가치, 로그 확률, 및 엔트로피를 계산한다.
        """

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        
        """주어진 관찰에 대한 행동을 결정한다.

        이 메소드는 액터 네트워크를 사용하여 주어진 관찰과 순환 신경망(RNN) 상태를 기반으로 행동을 결정한다.
        필요에 따라 결정적 또는 확률적 방식으로 행동을 선택할 수 있으며, 순환 정책을 사용하는 경우 업데이트된 RNN 상태를 반환한다.

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

        이 메소드는 액터와 크리틱 네트워크의 학습을 준비하기 위해 호출된다.
        PyTorch의 `.train()` 메소드를 사용하여 네트워크를 학습 모드로 전환함으로써,
        학습 중에만 사용되는 기능들(예: 드롭아웃, 배치 정규화 등)이 활성화된다.

        반환값:
            없음.
        """

        self.actor.train()
        self.critic.train()

    def prep_rollout(self):

        """액터와 크리틱 네트워크를 평가 모드로 설정한다.

        이 메소드는 평가나 실제 운영 환경에서 액터와 크리틱 네트워크의 성능을 평가하기 위해 호출된다.
        PyTorch의 `.eval()` 메소드를 사용하여 네트워크를 평가 모드로 전환함으로써,
        학습 중에만 사용되는 기능들(예: 드롭아웃, 배치 정규화 등)이 비활성화되고,
        모델의 추론 성능이 일관되게 유지된다.

        반환값:
            없음.
        """

        self.actor.eval()
        self.critic.eval()

    def copy(self):

        """현재 PPOPolicy 인스턴스의 복사본을 생성하여 반환한다.

        이 메소드는 현재 정책 인스턴스의 설정, 관찰 공간, 행동 공간, 그리고 계산 디바이스를
        그대로 사용하여 새로운 PPOPolicy 인스턴스를 생성한다. 이는 현재 정책의 상태를
        저장하거나, 실험에서 다양한 설정을 테스트하기 위해 유용하게 사용될 수 있다.

        반환값:
            PPOPolicy: 현재 인스턴스의 설정을 반영한 새 PPOPolicy 객체.

        예시:
            >>> new_policy = current_policy.copy()
            이 코드는 현재 PPOPolicy 인스턴스의 복사본을 생성하여 `new_policy`에 할당한다.
        """
        
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)
