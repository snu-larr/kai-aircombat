import gym.spaces
import numpy as np
from collections import OrderedDict


def build_flattener(space):

    """주어진 Gym 환경 공간에 적합한 Flattener 객체를 생성하여 반환한다.

    OpenAI Gym의 환경 공간을 효율적으로 처리하기 위해, 이 함수는 주어진 공간의 타입에 따라
    적절한 Flattener 객체를 생성하고 반환한다. 이 객체들은 공간을 일련의 숫자로 평탄화(flatten)하여
    학습 알고리즘에서 사용하기 용이하게 만든다.

    매개변수:
        space (gym.Space): OpenAI Gym 환경의 공간. Dict, Box, MultiDiscrete, Discrete 타입이 지원된다.

    반환값:
        Flattener: 해당 공간을 평탄화할 수 있는 Flattener 객체.

    예외:
        NotImplementedError: 주어진 공간 타입이 지원되지 않는 경우 발생.

    사용 예시:
        >>> space = gym.spaces.Box(low=0, high=1, shape=(3, 4))
        >>> flattener = build_flattener(space)
        >>> print(flattener)
        BoxFlattener(...)
    """

    if isinstance(space, gym.spaces.Dict):
        return DictFlattener(space)
    elif isinstance(space, gym.spaces.Box) \
            or isinstance(space, gym.spaces.MultiDiscrete):
        return BoxFlattener(space)
    elif isinstance(space, gym.spaces.Discrete):
        return DiscreteFlattener(space)
    else:
        raise NotImplementedError


class DictFlattener():

    def __init__(self, ori_space):

        """DictFlattener 클래스의 생성자.

        주어진 gym.spaces.Dict 공간을 평탄화하기 위한 객체를 초기화한다.
        각 하위 공간에 대해 적절한 Flattener 객체를 생성하고 관리한다.

        매개변수:
            ori_space (gym.spaces.Dict): 평탄화할 원본 gym.spaces.Dict 공간.

        속성:
            space (gym.spaces.Dict): 평탄화 대상의 원본 공간.
            size (int): 평탄화된 공간의 총 크기.
            flatteners (OrderedDict): 각 하위 공간의 이름을 키로, 해당 공간을 평탄화하는 Flattener 객체를 값으로 하는 정렬된 딕셔너리.

        예외:
            AssertionError: ori_space가 gym.spaces.Dict 타입이 아닐 때 발생.

        사용 예시:
            >>> ori_space = gym.spaces.Dict({
                    "position": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
                    "velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
                })
            >>> flattener = DictFlattener(ori_space)
            >>> print(flattener.size)
            6
        """
        
        self.space = ori_space
        assert isinstance(ori_space, gym.spaces.Dict)
        self.size = 0
        self.flatteners = OrderedDict()
        for name, space in self.space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                flattener = BoxFlattener(space)
            elif isinstance(space, gym.spaces.Discrete):
                flattener = DiscreteFlattener(space)
            elif isinstance(space, gym.spaces.Dict):
                flattener = DictFlattener(space)
            self.flatteners[name] = flattener
            self.size += flattener.size

    def __call__(self, observation):
        
        """주어진 관측값을 평탄화하여 numpy 배열로 변환한다.

        `observation`은 `OrderedDict` 형태이며, 각 키는 하위 공간의 이름과 일치해야 한다.
        이 메소드는 각 하위 공간의 관측값을 평탄화하고, 이를 하나의 numpy 배열로 합친다.

        매개변수:
            observation (OrderedDict): 평탄화할 관측값. 각 키는 하위 공간의 이름에 해당한다.

        반환값:
            numpy.ndarray: 평탄화된 관측값을 포함하는 numpy 배열.

        예외:
            AssertionError: observation이 OrderedDict 타입이 아닐 때 발생.

        사용 예시:
            >>> observation = OrderedDict({
                    "position": np.array([0.1, 0.2, 0.3]),
                    "velocity": np.array([-0.1, -0.2, -0.3]),
                })
            >>> flattener = DictFlattener(ori_space)
            >>> flat_observation = flattener(observation)
            >>> print(flat_observation.shape)
            (6,)
        """
        
        assert isinstance(observation, OrderedDict)
        batch = self.get_batch(observation, self)
        if batch == 1:
            array = np.zeros(self.size,)
        else:
            array = np.zeros(self.size)

        self.write(observation, array, 0)
        return array

    def inv(self, observation):

        """평탄화된 numpy 배열을 원래의 OrderedDict 관측값으로 디코딩한다.

        이 메소드는 평탄화된 numpy 배열을 받아 원래의 관측값 구조(`OrderedDict`)로 복원한다.
        각 하위 공간에 대한 평탄화를 역으로 수행하여, 원래 공간의 데이터 구조로 변환한다.

        매개변수:
            observation (numpy.ndarray): 디코딩할 평탄화된 numpy 배열.

        반환값:
            OrderedDict: 디코딩된 관측값을 포함하는 `OrderedDict` 객체.

        사용 예시:
            >>> flat_observation = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
            >>> flattener = DictFlattener(ori_space)
            >>> observation = flattener.inv(flat_observation)
            >>> print(observation)
            OrderedDict([('position', array([0.1, 0.2, 0.3])), ('velocity', array([-0.1, -0.2, -0.3]))])
        """
        
        offset_start, offset_end = 0, 0
        output = OrderedDict()
        for n, f in self.flatteners.items():
            offset_end += f.size
            output[n] = f.inv(observation[..., offset_start:offset_end])
            offset_start = offset_end
        return output

    def write(self, observation, array, offset):

        """주어진 OrderedDict 관측값을 numpy 배열에 평탄화하여 기록한다.

        각 하위 공간의 관측값을 해당하는 Flattener 객체를 사용하여 평탄화하고,
        주어진 numpy 배열의 지정된 위치부터 순차적으로 기록한다.

        매개변수:
            observation (OrderedDict): 평탄화할 관측값을 포함하는 `OrderedDict` 객체.
            array (numpy.ndarray): 평탄화된 데이터를 기록할 numpy 배열.
            offset (int): `array` 내에서 기록을 시작할 위치의 인덱스.

        사용 예시:
            >>> observation = OrderedDict([('position', np.array([0.1, 0.2, 0.3])), ('velocity', np.array([-0.1, -0.2, -0.3]))])
            >>> array = np.zeros(6)
            >>> flattener = DictFlattener(ori_space)
            >>> flattener.write(observation, array, 0)
            >>> print(array)
            [ 0.1  0.2  0.3 -0.1 -0.2 -0.3 ]
        이 메소드는 `observation`에 있는 각 하위 관측값을 평탄화하여 `array`에 기록한다.
        """
        
        for o, f in zip(observation.values(), self.flatteners.values()):
            f.write(o, array, offset)
            offset += f.size

    def get_batch(self, observation, flattener):

        """주어진 관측값의 배치 크기를 결정한다.

        관측값이 딕셔너리인 경우, 재귀적으로 첫 번째 하위 요소의 배치 크기를 반환한다.
        numpy 배열이나 리스트와 같은 형태인 경우, 해당 데이터의 크기를 평탄화할 공간의 크기로 나누어 배치 크기를 계산한다.

        매개변수:
            observation (dict 또는 numpy.ndarray): 배치 크기를 결정할 관측값.
            flattener (Flattener): 현재 처리 중인 평탄화 객체.

        반환값:
            int: 계산된 배치 크기.

        사용 예시:
            >>> observation = {
                    'position': np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
                    'velocity': np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
                }
            >>> flattener = DictFlattener(ori_space)
            >>> batch_size = flattener.get_batch(observation, flattener)
            >>> print(batch_size)
            2
        이 메소드는 주어진 관측값의 구조를 기반으로 배치 크기를 계산한다.
        """
        
        if isinstance(observation, dict):
            # 如果是字典的话返回第一个的batch
            for o, f in zip(observation.values(), flattener.flatteners.values()):
                return self.get_batch(o, f)
        else:
            return np.asarray(observation).size // flattener.size


class BoxFlattener():

    def __init__(self, ori_space):

        """BoxFlattener 클래스의 생성자.

        주어진 gym.spaces.Box 또는 gym.spaces.MultiDiscrete 공간을 평탄화하기 위한 객체를 초기화한다.
        이 클래스는 공간의 차원을 기반으로 평탄화된 데이터의 크기를 계산한다.

        매개변수:
            ori_space (gym.spaces.Box or gym.spaces.MultiDiscrete): 평탄화할 원본 공간.

        속성:
            space (gym.spaces.Box or gym.spaces.MultiDiscrete): 평탄화 대상의 원본 공간.
            size (int): 평탄화된 데이터의 총 크기.

        예외:
            AssertionError: ori_space가 gym.spaces.Box 또는 gym.spaces.MultiDiscrete 타입이 아닐 때 발생.

        사용 예시:
            >>> ori_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
            >>> flattener = BoxFlattener(ori_space)
            >>> print(flattener.size)
            3
        """
        
        self.space = ori_space
        assert isinstance(ori_space, gym.spaces.Box) \
            or isinstance(ori_space, gym.spaces.MultiDiscrete)
        self.size = np.product(ori_space.shape)

    def __call__(self, observation):

        """주어진 관측값을 numpy 배열로 변환하고 평탄화한다.

        단일 관측값 또는 배치 형태의 관측값을 처리할 수 있으며, 
        결과는 항상 평탄화된 형태로 반환된다.

        매개변수:
            observation: 평탄화할 관측값. numpy 배열 또는 리스트 형태가 가능하다.

        반환값:
            numpy.ndarray: 평탄화된 관측값을 포함하는 numpy 배열. 단일 관측값인 경우 1차원 배열,
                        배치 형태인 경우 2차원 배열([배치 크기, 평탄화된 크기])로 반환된다.

        사용 예시:
            >>> ori_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
            >>> flattener = BoxFlattener(ori_space)
            >>> observation = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
            >>> flat_obs = flattener(observation)
            >>> print(flat_obs.shape)
            (2, 3)
        """
        
        array = np.array(observation, copy=False)
        if array.size // self.size == 1:
            return array.ravel()
        else:
            return array.reshape(-1, self.size)

    def inv(self, observation):

        """평탄화된 numpy 배열을 원래의 공간의 형태로 복원한다.

        입력으로 주어진 평탄화된 관측값(numpy 배열)을 `self.space`의 원래 차원으로 다시 변환한다.
        단일 관측값과 배치 형태의 관측값 모두 처리할 수 있다.

        매개변수:
            observation (numpy.ndarray): 복원할 평탄화된 관측값.

        반환값:
            numpy.ndarray: 원래 공간의 형태로 복원된 관측값. 단일 관측값인 경우 원래 `self.space.shape`의 형태를 가지며,
                        배치 형태인 경우 첫 번째 차원이 배치 크기를 나타내는 2차원 배열로 반환된다.

        사용 예시:
            >>> ori_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
            >>> flattener = BoxFlattener(ori_space)
            >>> flat_obs = np.array([0.1, 0.2, 0.3])
            >>> observation = flattener.inv(flat_obs)
            >>> print(observation.shape)
            (3,)
        """
        
        array = np.array(observation, copy=False)
        if array.size // self.size == 1:
            return array.reshape(self.space.shape)
        else:
            return array.reshape((-1,) + self.space.shape)

    def write(self, observation, array, offset):
        """주어진 관측값을 평탄화하여 numpy 배열의 지정된 위치에 기록한다.

        `observation`을 평탄화한 후, 결과를 `array`의 `offset`부터 시작하는 위치에 기록한다.
        이 메소드는 `__call__`을 사용하여 관측값을 평탄화하며, 평탄화된 데이터를 `array`에 채운다.

        매개변수:
            observation: 평탄화할 관측값. numpy 배열 또는 그와 호환 가능한 데이터 구조.
            array (numpy.ndarray): 평탄화된 데이터를 기록할 대상 numpy 배열.
            offset (int): `array` 내에서 평탄화된 데이터를 기록하기 시작할 위치의 인덱스.

        사용 예시:
            >>> ori_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
            >>> flattener = BoxFlattener(ori_space)
            >>> array = np.zeros(3)
            >>> observation = np.array([0.1, 0.2, 0.3])
            >>> flattener.write(observation, array, 0)
            >>> print(array)
            [0.1 0.2 0.3]
        이 메소드는 주어진 `observation`을 `array`의 지정된 `offset` 위치부터 평탄화하여 기록한다.
        """
        array[..., offset:offset + self.size] = self(observation)


class DiscreteFlattener():
 
    def __init__(self, ori_space):

        """DiscreteFlattener 클래스의 생성자.

        주어진 gym.spaces.Discrete 공간을 평탄화하기 위한 객체를 초기화한다. 
        이 클래스는 이산 공간의 값들을 적절히 처리할 수 있도록 설정된다.

        매개변수:
            ori_space (gym.spaces.Discrete): 평탄화할 원본 gym.spaces.Discrete 공간.

        속성:
            space (gym.spaces.Discrete): 평탄화 대상의 원본 공간.
            size (int): 평탄화된 데이터의 크기. 이산 공간의 경우, 이 값은 항상 1이다.

        예외:
            AssertionError: ori_space가 gym.spaces.Discrete 타입이 아닐 때 발생.

        사용 예시:
            >>> ori_space = gym.spaces.Discrete(5)
            >>> flattener = DiscreteFlattener(ori_space)
            >>> print(flattener.size)
            1
        이 클래스는 주어진 Discrete 공간을 처리하기 위해 초기화되며, 평탄화 과정에서 생성되는 데이터의 크기는 1이다.
        """
        
        self.space = ori_space
        assert isinstance(ori_space, gym.spaces.Discrete)
        self.size = 1

    def __call__(self, observation):

        """주어진 이산형 관측값을 처리하여 스칼라 값 또는 열 벡터 형태로 변환한다.

        단일 관측값의 경우 스칼라 값으로 변환하여 반환하며, 배치 형태의 관측값인 경우
        각 관측값을 열 벡터 형태로 변환하여 반환한다.

        매개변수:
            observation: 처리할 이산형 관측값. 스칼라 값, numpy 배열 또는 리스트 형태가 가능하다.

        반환값:
            스칼라 값 또는 numpy.ndarray: 단일 관측값인 경우 스칼라 값으로 반환되며,
                                        배치 형태의 관측값인 경우 열 벡터 형태([n, 1])로 변환된 numpy 배열로 반환된다.

        사용 예시:
            >>> flattener = DiscreteFlattener(gym.spaces.Discrete(5))
            >>> observation = 3
            >>> flat_obs = flattener(observation)
            >>> print(flat_obs)
            3
            >>> batch_observation = [3, 1, 4]
            >>> flat_batch_obs = flattener(batch_observation)
            >>> print(flat_batch_obs)
            [[3]
            [1]
            [4]]
        """
        
        array = np.array(observation, copy=False)
        if array.size == 1:
            return array.item()
        else:
            return array.reshape(-1, 1)

    def inv(self, observation):

        """평탄화된 이산형 관측값을 원래의 형태로 복원한다.

        입력으로 주어진 평탄화된 관측값을 스칼라 값 또는 열 벡터 형태로 복원한다. 단일 관측값의 경우
        스칼라 값으로 반환되며, 배치 형태의 관측값인 경우 열 벡터 형태로 변환하여 반환한다.

        매개변수:
            observation: 복원할 평탄화된 관측값. 스칼라 값, numpy 배열 또는 리스트 형태가 가능하다.

        반환값:
            스칼라 값 또는 numpy.ndarray: 단일 관측값인 경우 스칼라 값으로 반환되며,
                                        배치 형태의 관측값인 경우 열 벡터 형태([n, 1])로 변환된 numpy 배열로 반환된다.

        사용 예시:
            >>> flattener = DiscreteFlattener(gym.spaces.Discrete(5))
            >>> flat_obs = 3
            >>> observation = flattener.inv(flat_obs)
            >>> print(observation)
            3
            >>> flat_batch_obs = np.array([[3], [1], [4]])
            >>> batch_observation = flattener.inv(flat_batch_obs)
            >>> print(batch_observation)
            [[3]
            [1]
            [4]]
        """
        
        array = np.array(observation, dtype=np.int, copy=False)
        if array.size == 1:
            return array.item()
        else:
            return array.reshape(-1, 1)

    def write(self, observation, array, offset):

        """주어진 이산형 관측값을 평탄화하여 numpy 배열의 지정된 위치에 기록한다.

        `observation`을 처리(평탄화)한 후, 그 결과를 `array`의 `offset`으로 지정된 위치에 기록한다.
        이 메소드는 이산 공간의 관측값을 평탄화하는 과정을 거치며, 이산 공간의 특성상 평탄화된
        데이터의 크기는 1이다.

        매개변수:
            observation: 평탄화할 이산형 관측값.
            array (numpy.ndarray): 평탄화된 데이터를 기록할 대상 numpy 배열.
            offset (int): `array` 내에서 평탄화된 데이터를 기록하기 시작할 위치의 인덱스.

        사용 예시:
            >>> flattener = DiscreteFlattener(gym.spaces.Discrete(5))
            >>> array = np.zeros(5)
            >>> observation = 3
            >>> flattener.write(observation, array, 2)
            >>> print(array)
            [0. 0. 3. 0. 0.]
        이 메소드는 주어진 `observation`을 처리하여 `array`의 지정된 `offset` 위치에 평탄화하여 기록한다.
        """
        
        array[..., offset:offset + 1] = self(observation)
