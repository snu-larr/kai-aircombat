# Air Combat RL Project with ACAM
규칙기반 AI 가상항공기 (ACAM from ARES) 를 이용하여 강화학습을 할 수 있는 학습 환경 프로그램이다.

## Definition
`Air Combat Effectiveness Analysis Model (ACAM)` : 아레스에서 개발한 항공무장효과분석모델

`학습기반 AI 가상항공기` : 강화학습을 통해 비행제어를 하는 가상항공기

`규칙기반 AI 가상항공기` : 규칙기반 (ACAM) 을 통해 비행제어를 하는 가상항공기

`(MA)PPO` : (Multi Agent) Proximal Policy Optimization

`JSBSim` : 오픈소스 비행 동역학 프로그램 [JSBSim Link](https://github.com/JSBSim-Team/jsbsim)

`TacView` : 오픈소스 비행 데이터 분석 툴 [TacView Link](https://www.tacview.net/)

## Install 

```shell
# create python env
conda create -n jsbsim python=3.8
# install dependency
pip install numpy torch matplotlib gym tensorboard pymap3d jsbsim==1.2.0 geographiclib icecream setproctitle shapely

- Initialize submodules(*JSBSim-Team/jsbsim*): `git submodule init; git submodule update`
```

## Architecture
`envs/JSBSim/envs/env_base.py` 코드 내에 ACAM 과 송수신한 데이터를 parsing 하는 부분이 있다. 해당 부분의 `socket_send_recv` 함수에 ACAM 과 송수신한 데이터를 활용할 수 있다.

`algorithms/ppo/ppo_*.py` 코드 내에서 PPO의 actor, critic 신경망 구조를 설계하고 Advantage 를 구하여 Policy, Value Loss 를 구하는 함수로 구성되어 있다.

`envs/JSBSim/core/simulator.py` 코드 내에 항공기 객체 정보가 들어있으며 JSBSim 과 연동되어 동역학 물리량을 받는다. 해당 폴더내의 `catalog.py` 코드는 JSBSim 과 연동되는 property 를 정의한 파일이다.

`envs/JSBSim/reward_functions/` 폴더는 보상함수로 사용되는 항목들을 나열하였다. 고도, 방향, 임무성공/실패 등의 항목으로 구성되어있다.

`envs/JSBSim/tasks/` 폴더는 시나리오 별 임무를 나열하여 사용하는 신경망 구조, 보상함수, 종료조건 등을 지정하는 폴더이다.

`envs/JSBSim/termination_conditions/` 폴더는 종료조건함수로 사용되는 항목들을 나열하였다. 시간초과, 전멸/격추 등의 항목으로 구성되어있다.

`envs/env_wrappers.py` Pipe 형태의 IPC 를 이용하여 multi processing 을 가능하게 하는 코드이다.

`runner/` 폴더는 가상 전장 환경과 학습기반 AI 가상항공기가 학습할 수 있게 실행하여 경험을 쌓는 부분, 쌓은 경험을 바탕으로 학습을 진행할 수 있도록 하는 부분이 담긴 코드이다.

`scripts/` 폴더는 학습, 운용할 실행파일을 bash 파일 형태로 만들어 놓은 폴더이다.

`scripts/ares_test_code.cpp` ACAM 없이 학습 코드를 실행할 경우, dummy data 로 규칙기반 AI 가상항공기 정보를 전달해주는 test code 이다.

## Neural Network Architecture
신경망의 경우 MLP-GRU-MLP 의 가장 간단한 형태로 구성되어 있다.
해당 신경망 정보는 `algorithms/utils/ppo_actor.py` 코드를 확인하면 된다.


## Communication Flow
초기화 시 ACAM 에서 시나리오 설정에 관련된 데이터 파일을 송신하면, 해당 데이터를 이용하여 정보를 수신한다. 수신 된 정보를 이용하여 AI 내 가상 전장환경을 구성하고, 학습을 시작할 수 있도록 전투기, 무장, 지대공 위협의 객체를 생성한다.

만들어진 객체 정보를 이용하여 t초의 상황을 이용해 AI 가상항공기부터 t+1 상황과 신경망 출력 (무장 trigger) 값을 ACAM 에 전달하고, 무장 trigger 정보를 ACAM 이 반영하여 t+1 의 규칙기반 가상항공기 정보와 무장 정보를 AI 에 송신한다.

##### ARES 와의 통신 구조
![ARES 와의 통신 구조](assets/ARES%20와%20통신%20구조.png)


##### 초기화 진행 시 초기화 방법
![초기화 진행 시 초기화 방법](assets/초기화%20진행%20시%20초기화%20방법.png)

##### 종료조건 시 초기화 방법
![종료조건 시 초기화 방법](assets/종료조건%20시%20초기화%20방법.png)


## Envs
`envs/JSBSim/configs` 안에 각 시나리오 별 임무 계획이 있으며, 시나리오 별 임무를 추가할 수 있다. 현재 있는 임무로는 SAM, 1v1, 2v2 로 구성되어있다.

### SingleCombat
SAM task 와 1v1 공중 교전 임무로 구성되어 있다. AI 내에 무장 라이브러리를 사용하지 않기 때문에, 무장 발사 trigger 를 ACAM 에 전달하는 방식으로 진행된다.


### MultiCombat
2v2 공중 교전 임무로 구성되어 있다. AI 내에 무장 라이브러리를 사용하지 않기 때문에, 무장 발사 trigger 를 ACAM 에 전달하는 방식으로 진행된다.

## Quick Start
### Training

```bash
# 1. run cpp code (scripts/ares_test_code.cpp)

# 2. run AI code
cd scripts
bash train_*.sh
```

Visual Studio 2019 를 이용하여 ares_test_code.cpp 를 실행 (굳이 Visual Studio 를 쓰지 않아도 되며 각자 지닌 cpp compiler 를 이용해도 됨)

- `--env-name` 가상 전장 환경 정보로 ['SingleCombat', 'MultipleCombat'] 를 사용할 수 있다.
- `--scenario` `envs/JBSim/configs` 에 정의된 yaml 파일로 진행될 수 있다.
- `--algorithm` PPO/MAPPO 를 선택적으로 사용할 수 있다.

추가적인 hyper parameter 는 `train_*.sh` 혹은 `config.py` 를 참고하여 수정 및 사용할 수 있다.

### Evaluate and Render
```bash
# 1. run cpp code (scripts/ares_test_code.cpp)

# 2. run AI code
cd envs/JSBSim/test
python test_*.py
```

저장되는 acmi 파일을 이용하여 Tacview 를 통해 화면에 결과를 출력한다. 


## Reference
참조 github repo : https://github.com/liuqh16/CloseAirCombat