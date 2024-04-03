import torch
import torch.nn as nn
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm


class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        """PPOTrainer 클래스의 인스턴스를 초기화한다.

        PPO 알고리즘의 학습을 위한 여러 설정을 초기화하며, 이 설정들은 PPO 학습 과정의
        다양한 측면을 제어한다. 이러한 설정에는 에폭 수, 클리핑 매개변수, 가치 손실 계수,
        엔트로피 계수 등이 포함된다.

        매개변수:
            args (Namespace): 학습 설정을 포함하는 객체. PPO 알고리즘의 다양한 매개변수 및
                            학습 과정에서 사용되는 설정이 포함될 수 있다.
            device (torch.device, optional): 계산에 사용할 디바이스(cpu 또는 cuda). 기본값은 'cpu'.

        속성:
            device: 계산에 사용될 장치.
            tpdv: 텐서의 데이터 타입과 장치를 지정하는 딕셔너리.
            ppo_epoch: PPO 업데이트를 실행할 에폭 수.
            clip_param: PPO 클리핑 매개변수의 값.
            use_clipped_value_loss: 클리핑된 가치 손실을 사용할지 여부.
            num_mini_batch: 미니 배치의 수.
            value_loss_coef: 가치 손실 계수.
            entropy_coef: 엔트로피 계수.
            use_max_grad_norm: 최대 그래디언트 노름을 사용할지 여부.
            max_grad_norm: 사용될 최대 그래디언트 노름.
            use_recurrent_policy: 순환 정책을 사용할지 여부.
            data_chunk_length: 데이터 청크의 길이(순환 정책 사용 시).

        예시:
            >>> args = Namespace(ppo_epoch=4, clip_param=0.2, ...)
            >>> trainer = PPOTrainer(args)
            이 코드는 주어진 설정으로 PPOTrainer 인스턴스를 초기화한다.
        """

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

    def ppo_update(self, policy: PPOPolicy, sample):

        """PPO 알고리즘을 사용하여 정책을 업데이트한다.

        주어진 샘플 데이터를 바탕으로 PPO 정책의 손실을 계산하고, 이를 최적화하여
        정책 매개변수를 업데이트한다. 손실 계산에는 정책 손실, 가치 손실,
        그리고 엔트로피 손실이 포함된다.

        매개변수:
            policy (PPOPolicy): 업데이트할 정책 객체.
            sample (tuple): 학습에 사용할 샘플 데이터. 여러 텐서(obs_batch, actions_batch 등)를 포함한다.

        반환:
            policy_loss (Tensor): 계산된 정책 손실.
            value_loss (Tensor): 계산된 가치 손실.
            policy_entropy_loss (Tensor): 계산된 정책의 엔트로피 손실.
            ratio (Tensor): 정책 업데이트 비율.
            actor_grad_norm (float): 액터 네트워크의 그래디언트 노름.
            critic_grad_norm (float): 크리틱 네트워크의 그래디언트 노름.

        예시:
            >>> policy = PPOPolicy(args, ...)
            >>> sample = ...
            >>> policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)
            이 코드는 주어진 정책과 샘플 데이터를 사용하여 PPO 업데이트를 수행한다.
        """

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = policy.evaluate_actions(obs_batch,
                                                                         rnn_states_actor_batch,
                                                                         rnn_states_critic_batch,
                                                                         actions_batch,
                                                                         masks_batch)

        # Obtain the loss function
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # Optimize the loss function
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):

        """PPOPolicy를 사용하여 주어진 데이터 버퍼로부터 정책을 학습시킨다.

        이 메소드는 주어진 PPOPolicy 인스턴스를 사용하여, ReplayBuffer에서 제공하는 샘플 데이터를 바탕으로
        정책을 여러 에폭 동안 학습시킨다. 학습 과정에서는 정책 손실, 가치 손실, 정책의 엔트로피 손실 등을
        계산하고 최적화한다.

        매개변수:
            policy (PPOPolicy): 학습시킬 정책 객체.
            buffer (Union[ReplayBuffer, List[ReplayBuffer]]): 학습 데이터를 제공하는 버퍼 또는 버퍼의 리스트.

        반환:
            train_info (dict): 학습 과정에서 계산된 메트릭을 담은 딕셔너리. 가치 손실, 정책 손실,
                            정책의 엔트로피 손실, 액터와 크리틱의 그래디언트 노름, 업데이트 비율을 포함한다.

        예시:
            >>> policy = PPOPolicy(args, ...)
            >>> buffer = ReplayBuffer(...)
            >>> train_info = trainer.train(policy, buffer)
            이 코드는 주어진 정책과 버퍼를 사용하여 PPO 알고리즘에 따라 정책을 학습시킨다.
        """
        
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
