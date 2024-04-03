import torch
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod
from .utils import get_shape_from_space


class Buffer(ABC):

    def __init__(self):
        """Buffer 클래스의 인스턴스를 초기화한다.

        이 클래스는 강화 학습에서 사용되는 데이터 버퍼의 인터페이스를 정의하는 추상 기본 클래스이다.
        실제 구현은 이 클래스를 상속받는 하위 클래스에서 제공된다.

        반환값: 없음.
        """
        pass

    @abstractmethod
    def insert(self, **kwargs):
        """Buffer에 데이터를 삽입한다.

        이 메소드는 버퍼에 새로운 데이터를 삽입하는 추상 메소드로, 구체적인 구현은 하위 클래스에서 제공된다.

        매개변수:
            **kwargs: 삽입할 데이터의 키워드 인자.

        반환값: 없음.
        """
        pass

    @abstractmethod
    def after_update(self):
        """Buffer 업데이트 후 처리를 수행한다.

        이 메소드는 버퍼가 업데이트된 후에 필요한 후처리 작업을 수행하는 추상 메소드로,
        구체적인 구현은 하위 클래스에서 제공된다.

        반환값: 없음.
        """
        pass

    @abstractmethod
    def clear(self):
        """Buffer의 내용을 초기화한다.

        이 메소드는 버퍼의 모든 데이터를 삭제하고 초기 상태로 되돌리는 추상 메소드로,
        구체적인 구현은 하위 클래스에서 제공된다.

        반환값: 없음.
        """
        pass


class ReplayBuffer(Buffer):

    @staticmethod
    def _flatten(T: int, N: int, x: np.ndarray):

        """주어진 배열을 (T * N, ...) 형태로 평탄화한다.

        강화 학습에서 여러 에피소드와 에이전트의 데이터를 하나의 배치로 결합할 때 사용되는 유틸리티 함수입니다.
        시간 차원(T)과 에이전트 차원(N)을 합쳐서, 데이터를 더 쉽게 처리할 수 있도록 배열의 형태를 변경합니다.

        매개변수:
            T (int): 시간 차원의 크기.
            N (int): 에이전트 차원의 크기.
            x (np.ndarray): 평탄화할 다차원 배열.

        반환값:
            np.ndarray: 평탄화된 배열. 원래 배열의 첫 두 차원이 합쳐진 형태로, 나머지 차원은 유지된다.

        예시:
            >>> T, N = 5, 3
            >>> x = np.random.rand(T, N, 2, 2)
            >>> flattened_x = _flatten(T, N, x)
            >>> print(flattened_x.shape)
            (15, 2, 2)
        """

        return x.reshape(T * N, *x.shape[2:])

    @staticmethod
    def _cast(x: np.ndarray):

        """주어진 배열의 차원을 재배열하고 필요한 경우 평탄화한다.

        이 메소드는 강화 학습 데이터를 모델에 적합한 형태로 변환할 때 사용된다. 특히, 시간 차원과 배치 차원을
        재배열하여 순환 신경망과 같은 모델 구조에 적합한 형태로 만든다.

        매개변수:
            x (np.ndarray): 차원을 재배열할 다차원 배열.

        반환값:
            np.ndarray: 재배열 및 평탄화된 배열. 첫 번째 차원이 시간 차원과 배치 차원의 조합으로 재구성되며,
                        나머지 차원은 원본 배열의 차원 순서를 유지하면서 평탄화된다.

        예시:
            >>> x = np.random.rand(5, 3, 2, 2)  # 예: 시간 차원(T=5), 배치 차원(N=3), 나머지 차원
            >>> casted_x = _cast(x)
            >>> print(casted_x.shape)
            (15, 2, 2)  # 첫 번째 차원은 시간과 배치 차원이 결합되어 평탄화되고, 나머지 차원은 유지된다.
        """

        return x.transpose(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])

    def __init__(self, args, num_agents, obs_space, act_space):

        """ReplayBuffer 클래스의 인스턴스를 초기화하고, 강화 학습 에이전트의 경험을 저장하기 위한 데이터 구조를 설정한다.

        매개변수:
            args: 버퍼 및 학습 관련 설정을 포함하는 객체.
            num_agents (int): 에이전트의 수.
            obs_space: 관찰 공간을 정의하는 객체. 이는 환경으로부터의 관찰 데이터 형태를 결정한다.
            act_space: 행동 공간을 정의하는 객체. 이는 가능한 행동의 종류와 범위를 결정한다.

        속성:
            buffer_size (int): 버퍼에 저장할 경험의 최대 시간 단계 수.
            n_rollout_threads (int): 동시에 실행될 롤아웃 스레드의 수.
            num_agents (int): 에이전트의 수.
            gamma (float): 할인 계수.
            use_proper_time_limits (bool): 적절한 시간 제한 사용 여부.
            use_gae (bool): 일반화된 가치 추정 사용 여부.
            gae_lambda (float): GAE 람다.
            recurrent_hidden_size (int): 순환 신경망 은닉층의 크기.
            recurrent_hidden_layers (int): 순환 신경망 은닉층의 수.
            obs (np.ndarray): 관찰 데이터를 저장하는 배열.
            actions (np.ndarray): 행동 데이터를 저장하는 배열.
            rewards (np.ndarray): 보상 데이터를 저장하는 배열.
            masks (np.ndarray): 관찰이 종료 상태인지 나타내는 마스크.
            bad_masks (np.ndarray): 관찰이 실제 종료 상태인지 시간 제한으로 인한 종료 상태인지 나타내는 마스크.
            action_log_probs (np.ndarray): 선택된 행동의 로그 확률.
            value_preds (np.ndarray): 가치 예측값.
            returns (np.ndarray): 반환값.
            rnn_states_actor (np.ndarray): 액터의 순환 신경망 상태.
            rnn_states_critic (np.ndarray): 크리틱의 순환 신경망 상태.
            step (int): 현재 저장된 시간 단계의 인덱스.
        """

        # buffer config
        self.buffer_size = args.buffer_size
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.use_proper_time_limits = args.use_proper_time_limits
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, a_0, r_0, d_1, o_1, ... , d_T, o_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: bad_masks[t] = 'bad_transition' in info[t-1], which indicates whether obs[t] a true terminal state or time limit end state
        self.bad_masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    @property
    def advantages(self) -> np.ndarray:

        """에이전트의 행동이 가져다주는 기대 이익을 계산하고 표준화한 값을 반환한다.

        이 속성은 강화 학습에서 에이전트의 행동 선택이 가져오는 추가적인 가치를 평가하는 데 사용된다.
        계산된 이점은 반환값과 가치 예측값의 차이로 정의되며, 이 값은 표준화되어 반환된다.

        반환값:
            np.ndarray: 표준화된 이점 값. 이 값은 학습 과정에서 정책 그래디언트를 계산하는 데 사용될 수 있다.

        예시:
            >>> buffer = ReplayBuffer(args, num_agents, obs_space, act_space)
            >>> standardized_advantages = buffer.advantages
            이 코드는 `ReplayBuffer` 인스턴스에서 표준화된 이점 값을 계산한다.
        """

        advantages = self.returns[:-1] - self.value_preds[:-1]  # type: np.ndarray
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def insert(self,
               obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               **kwargs):
        
        """버퍼에 강화 학습 에이전트의 경험 데이터를 저장한다.

        주어진 관찰, 행동, 보상, 마스크, 행동의 로그 확률, 가치 예측, 순환 신경망 상태를 버퍼에 저장한다.
        선택적으로 '나쁜' 마스크 데이터도 저장할 수 있다.

        매개변수:
            obs (np.ndarray): 현재 시점의 관찰 데이터.
            actions (np.ndarray): 선택된 행동 데이터.
            rewards (np.ndarray): 받은 보상 데이터.
            masks (np.ndarray): 다음 시점이 종료 상태인지 나타내는 마스크 데이터.
            action_log_probs (np.ndarray): 선택된 행동의 로그 확률 데이터.
            value_preds (np.ndarray): 현재 시점의 가치 예측 데이터.
            rnn_states_actor (np.ndarray): 현재 시점의 액터 순환 신경망 상태.
            rnn_states_critic (np.ndarray): 현재 시점의 크리틱 순환 신경망 상태.
            bad_masks (Union[np.ndarray, None]): 환경에서 에피소드가 종료된 이유를 구분하는 '나쁜' 마스크 데이터. None일 경우 저장하지 않음.

        반환값: 없음.
        """

        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):

        """모델 업데이트가 완료된 후 버퍼의 마지막 시간 단계 데이터를 첫 번째 인덱스로 복사한다.

        이 메소드는 각 모델 업데이트 사이클 후에 호출되어, 롤아웃 중에 생성된 데이터 시퀀스의 연속성을 유지한다.
        마지막 상태, 마스크, 순환 신경망 상태들을 버퍼의 시작 부분으로 복사하여 다음 롤아웃의 초기 상태로 사용할 수 있게 준비한다.

        반환값: 없음.
        """   

        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()

    def clear(self):

        """버퍼의 모든 데이터를 초기화한다.

        새로운 에피소드 또는 학습 사이클을 시작하기 전에 버퍼를 재사용할 준비를 위해 호출된다.
        모든 데이터 배열을 초기 상태로 재설정하고, 스텝 카운터를 0으로 재설정한다.

        반환값: 없음.
        """

        self.step = 0
        self.obs = np.zeros_like(self.obs, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        self.bad_masks = np.ones_like(self.bad_masks, dtype=np.float32)
        self.action_log_probs = np.zeros_like(self.action_log_probs, dtype=np.float32)
        self.value_preds = np.zeros_like(self.value_preds, dtype=np.float32)
        self.returns = np.zeros_like(self.returns, dtype=np.float32)
        self.rnn_states_actor = np.zeros_like(self.rnn_states_critic)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

    def compute_returns(self, next_value: np.ndarray):

        """보상의 할인된 합을 계산하거나 GAE를 사용하여 반환값을 계산한다.

        마지막 에피소드 스텝 이후의 가치 예측을 기반으로 모든 시점에 대한 반환값을 계산한다.
        계산 방식은 적절한 시간 제한 사용 여부와 GAE 사용 여부에 따라 달라진다.

        매개변수:
            next_value (np.ndarray): 마지막 에피소드 스텝 이후의 가치 예측값.

        반환값: 없음.
        """

        if self.use_proper_time_limits:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) \
                        * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    @staticmethod
    def recurrent_generator(buffer: Union[Buffer, List[Buffer]], num_mini_batch: int, data_chunk_length: int):

        """순환 신경망 학습을 위한 청크별 미니 배치 데이터를 생성하는 생성기.

        이 생성기는 버퍼에 저장된 데이터를 시퀀스 청크로 나누고, 이를 미니 배치로 구성하여 순환 신경망 학습에 사용한다.
        데이터는 시퀀스에 따라 무작위로 섞이며, 각 미니 배치는 순차적으로 학습에 활용될 수 있다.

        매개변수:
            buffer (Union[Buffer, List[Buffer]]): 학습 데이터를 제공하는 버퍼 또는 버퍼의 리스트.
            num_mini_batch (int): 전체 배치를 몇 개의 미니 배치로 나눌지 결정하는 숫자.
            data_chunk_length (int): RNN 학습에 사용될 시퀀스 청크의 길이.

        반환값:
            Iterator[Tuple[np.ndarray, ...]]: 학습에 사용될 미니 배치를 순차적으로 반환하는 이터레이터. 각 미니 배치는
            관찰값, 행동, 마스크, 행동 로그 확률, 이득, 반환값, 가치 예측값, 액터 및 크리틱의 순환 신경망 상태를 포함한다.

        예시:
            >>> generator = ReplayBuffer.recurrent_generator(buffer, 4, 20)
            >>> for mini_batch in generator:
            >>>     obs_batch, actions_batch, ... = mini_batch
            이 코드는 주어진 버퍼로부터 순환 신경망 학습에 적합한 미니 배치를 생성한다.
        """

        buffer = [buffer] if isinstance(buffer, ReplayBuffer) else buffer  # type: List[ReplayBuffer]
        n_rollout_threads = buffer[0].n_rollout_threads
        buffer_size = buffer[0].buffer_size
        num_agents = buffer[0].num_agents
        assert all([b.n_rollout_threads == n_rollout_threads for b in buffer]) \
            and all([b.buffer_size == buffer_size for b in buffer]) \
            and all([b.num_agents == num_agents for b in buffer]) \
            and all([isinstance(b, ReplayBuffer) for b in buffer]), \
            "Input buffers must has the same type and shape"
        buffer_size = buffer_size * len(buffer)

        assert n_rollout_threads * buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) * num_agents ({})"
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, buffer_size, num_agents, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = np.vstack([ReplayBuffer._cast(buf.obs[:-1]) for buf in buffer])
        actions = np.vstack([ReplayBuffer._cast(buf.actions) for buf in buffer])
        masks = np.vstack([ReplayBuffer._cast(buf.masks[:-1]) for buf in buffer])
        old_action_log_probs = np.vstack([ReplayBuffer._cast(buf.action_log_probs) for buf in buffer])
        advantages = np.vstack([ReplayBuffer._cast(buf.advantages) for buf in buffer])
        returns = np.vstack([ReplayBuffer._cast(buf.returns[:-1]) for buf in buffer])
        value_preds = np.vstack([ReplayBuffer._cast(buf.value_preds[:-1]) for buf in buffer])
        rnn_states_actor = np.vstack([ReplayBuffer._cast(buf.rnn_states_actor[:-1]) for buf in buffer])
        rnn_states_critic = np.vstack([ReplayBuffer._cast(buf.rnn_states_critic[:-1]) for buf in buffer])

        # Get mini-batch size and shuffle chunk data
        data_chunks = n_rollout_threads * buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            actions_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *buffer[0].rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *buffer[0].rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = ReplayBuffer._flatten(L, N, obs_batch)
            actions_batch = ReplayBuffer._flatten(L, N, actions_batch)
            masks_batch = ReplayBuffer._flatten(L, N, masks_batch)
            old_action_log_probs_batch = ReplayBuffer._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = ReplayBuffer._flatten(L, N, advantages_batch)
            returns_batch = ReplayBuffer._flatten(L, N, returns_batch)
            value_preds_batch = ReplayBuffer._flatten(L, N, value_preds_batch)

            yield obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch


class SharedReplayBuffer(ReplayBuffer):

    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):

        """SharedReplayBuffer 클래스의 인스턴스를 초기화하고, 멀티 에이전트 강화 학습 데이터를 저장하기 위한 구조를 설정한다.

        매개변수:
            args: 학습 및 환경 설정을 포함하는 객체.
            num_agents (int): 환경 내 에이전트의 수.
            obs_space: 개별 에이전트의 관찰 공간을 정의하는 객체.
            share_obs_space: 모든 에이전트에 공유되는 관찰 공간을 정의하는 객체.
            act_space: 에이전트의 행동 공간을 정의하는 객체.

        속성:
            num_agents (int): 에이전트의 수.
            n_rollout_threads (int): 동시에 실행될 롤아웃 스레드의 수.
            gamma (float): 할인 계수.
            buffer_size (int): 버퍼에 저장할 최대 시간 단계의 수.
            use_proper_time_limits (bool): 적절한 시간 제한을 사용할지 여부.
            use_gae (bool): 일반화된 가치 추정(GAE)을 사용할지 여부.
            gae_lambda (float): GAE 람다 값.
            recurrent_hidden_size (int): 순환 신경망의 은닉층 크기.
            recurrent_hidden_layers (int): 순환 신경망의 은닉층 수.
            obs (np.ndarray): 개별 에이전트의 관찰 데이터.
            share_obs (np.ndarray): 모든 에이전트에 공유되는 관찰 데이터.
            actions (np.ndarray): 에이전트의 행동 데이터.
            rewards (np.ndarray): 에이전트의 보상 데이터.
            masks (np.ndarray): 시간 단계별 에피소드 종료 여부를 나타내는 마스크.
            bad_masks (np.ndarray): '나쁜' 종료(예: 시간 초과)를 나타내는 마스크.
            active_masks (np.ndarray): 각 시간 단계에서 각 에이전트의 활성화 상태를 나타내는 마스크.
            action_log_probs (np.ndarray): 선택된 행동의 로그 확률.
            value_preds (np.ndarray): 가치 예측값.
            returns (np.ndarray): 반환값.
            rnn_states_actor (np.ndarray): 액터 순환 신경망 상태.
            rnn_states_critic (np.ndarray): 크리틱 순환 신경망 상태.
            step (int): 현재 저장된 시간 단계의 인덱스.
        """

        # env config
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        # buffer config
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.use_proper_time_limits = args.use_proper_time_limits
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        share_obs_shape = get_shape_from_space(share_obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, s_0, a_0, r_0, d_0, ..., o_T, s_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state .... same for all agents
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        # NOTE: active_masks[t, :, i] represents whether agent[i] is alive in obs[t] .... differ in different agents
        self.active_masks = np.ones_like(self.masks)
        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    def insert(self,
               obs: np.ndarray,
               share_obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               active_masks: Union[np.ndarray, None] = None,
               available_actions: Union[np.ndarray, None] = None):
        
        """버퍼에 멀티 에이전트 강화 학습 데이터를 저장한다.

        각 시간 단계에서의 개별 및 공유된 관찰, 행동, 보상, 종료 신호, 행동의 로그 확률,
        가치 예측, 순환 신경망 상태를 저장한다. 선택적으로 에이전트별 활성화 상태와 가능한 행동도 저장할 수 있다.

        매개변수:
            obs (np.ndarray): 개별 에이전트 관찰 데이터.
            share_obs (np.ndarray): 모든 에이전트에 공유된 관찰 데이터.
            actions (np.ndarray): 선택된 행동 데이터.
            rewards (np.ndarray): 받은 보상 데이터.
            masks (np.ndarray): 다음 시점이 종료 상태인지 나타내는 마스크 데이터.
            action_log_probs (np.ndarray): 선택된 행동의 로그 확률 데이터.
            value_preds (np.ndarray): 현재 시점의 가치 예측 데이터.
            rnn_states_actor (np.ndarray): 액터 순환 신경망 상태.
            rnn_states_critic (np.ndarray): 크리틱 순환 신경망 상태.
            bad_masks (Union[np.ndarray, None]): '나쁜' 종료를 나타내는 마스크 데이터. None일 경우 저장하지 않음.
            active_masks (Union[np.ndarray, None]): 각 시간 단계에서 각 에이전트의 활성화 상태를 나타내는 마스크 데이터. None일 경우 저장하지 않음.
            available_actions (Union[np.ndarray, None]): 가능한 행동을 나타내는 데이터. None일 경우 저장하지 않음.

        반환값: 없음.
        """

        self.share_obs[self.step + 1] = share_obs.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            pass
        return super().insert(obs, actions, rewards, masks, action_log_probs, value_preds, rnn_states_actor, rnn_states_critic)

    def after_update(self):

        """모델 업데이트 후 버퍼의 데이터를 초기 위치로 복사하는 처리를 수행한다.

        이 메소드는 멀티 에이전트 설정에서의 공유된 관찰(`share_obs`)과 에이전트별 활성화 상태(`active_masks`)를
        다음 롤아웃이나 학습 사이클의 시작점으로 설정하기 위해 필요한 후처리를 수행한다.
        부모 클래스의 `after_update` 메소드를 호출하여 기본적인 후처리도 함께 수행한다.

        반환값: 없음.
        """

        self.active_masks[0] = self.active_masks[-1].copy()
        self.share_obs[0] = self.share_obs[-1].copy()
        return super().after_update()

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):

        """순환 신경망(RNN) 학습을 위한 시퀀스 데이터를 미니 배치로 생성하는 생성기.

        이 메소드는 멀티 에이전트 환경에서의 공유된 관찰 및 개별 에이전트 정보를 포함한 시퀀스 데이터를
        청크로 나누어 순차적인 학습에 적합한 형태로 제공한다. 데이터는 시퀀스에 따라 무작위로 섞이며,
        각 미니 배치는 순차적으로 학습에 활용될 수 있다.

        매개변수:
            advantages (np.ndarray): 이점 추정값.
            num_mini_batch (int): 전체 데이터를 몇 개의 미니 배치로 나눌지 결정하는 숫자.
            data_chunk_length (int): RNN 학습에 사용될 시퀀스 청크의 길이.

        반환값:
            Iterator[Tuple[np.ndarray, ...]]: 학습에 사용될 미니 배치를 순차적으로 반환하는 이터레이터. 각 미니 배치는
            개별 관찰, 공유된 관찰, 행동, 마스크, 활성화 마스크, 행동 로그 확률, 이점, 반환값, 가치 예측값,
            액터 및 크리틱의 순환 신경망 상태를 포함한다.

        예시:
            >>> advantages = np.random.rand(buffer_size, n_rollout_threads, num_agents)
            >>> generator = buffer.recurrent_generator(advantages, 4, 20)
            >>> for mini_batch in generator:
            >>>     obs_batch, share_obs_batch, ... = mini_batch
            이 코드는 주어진 이점 데이터를 기반으로 RNN 학습에 적합한 미니 배치를 생성한다.
        """

        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])
        active_masks = self._cast(self.active_masks[:-1])
        old_action_log_probs = self._cast(self.action_log_probs)
        advantages = self._cast(advantages)
        returns = self._cast(self.returns[:-1])
        value_preds = self._cast(self.value_preds[:-1])
        rnn_states_actor = self._cast(self.rnn_states_actor[:-1])
        rnn_states_critic = self._cast(self.rnn_states_critic[:-1])

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * self.buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = self._flatten(L, N, obs_batch)
            share_obs_batch = self._flatten(L, N, share_obs_batch)
            actions_batch = self._flatten(L, N, actions_batch)
            masks_batch = self._flatten(L, N, masks_batch)
            active_masks_batch = self._flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = self._flatten(L, N, advantages_batch)
            returns_batch = self._flatten(L, N, returns_batch)
            value_preds_batch = self._flatten(L, N, value_preds_batch)

            yield obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch
