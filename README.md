# UAV 5대 군집 유도 및 Independent TD3

고정된 목표점을 중심으로 UAV 5대가 선회하는 학습 환경이다. Python 3.10 이상, NumPy, PyTorch가 필요하다.

```bash
cd /home/laaa/td3_swarm_python
python3 train.py --episodes 10 --steps 2000
```

필요한 패키지가 없는 환경에서는 `python3 -m pip install -r requirements.txt`로 설치한다. 학습된 가중치는 기본적으로 `runs/td3_swarm.pt`에 저장된다. CPU를 지정하려면 `--device cpu`를 붙인다.

## 1. 처음 실행할 때

아래 명령은 CPU에서 1회, 32스텝만 돌려 설치와 학습이 정상인지 확인한다. 한 스텝은 시뮬레이션 시간 0.1초다.

```bash
cd /home/laaa/td3_swarm_python
python3 train.py --device cpu --episodes 1 --steps 32 --warmup 8 --batch-size 8 --policy-delay 2 --output runs/smoke.pt
```

터미널에 아래와 같은 결과가 나온다. 보상값은 실행 환경에 따라 달라질 수 있다.

```text
episode=1 steps=32 mean_return=-51.8846 updates_per_agent=25
Saved runs/smoke.pt
```

| 출력 | 의미 |
|---|---|
| `episode` | 방금 마친 에피소드 번호 |
| `steps` | 이번 학습 시작 이후 전체 누적 스텝 수 |
| `mean_return` | 해당 에피소드에서 UAV별로 합산한 보상을 5대에 대해 평균한 값. 같은 길이와 설정의 에피소드끼리 비교한다. |
| `updates_per_agent` | 에이전트별 누적 Critic 업데이트 수. Actor는 `--policy-delay` 간격으로 업데이트된다. |
| `Saved` | 학습 완료 후 가중치를 저장한 경로 |

본 학습에서 2,000스텝은 에피소드당 시뮬레이션 시간 200초다. 기본 출력 `runs/td3_swarm.pt`는 학습 종료 시 생성되며, 같은 경로로 다시 실행하면 덮어쓴다. 실행별로 보관하려면 `--output runs/원하는이름.pt`를 지정한다. 기본 장치 설정은 CUDA를 사용할 수 있으면 GPU, 그 외에는 CPU다.

`.pt`에는 UAV별 Actor 5개, UAV별 Critic 2개씩 총 10개의 가중치, 행동 순서, 관측 크기, 에피소드·스텝·업데이트 수가 저장된다. 현재 학습 실행의 출력은 터미널 요약과 이 가중치 파일이다. 그래프나 궤적 CSV를 생성하는 기능은 추가 구현이 필요하다.

## 2. 구현 위치와 실행 흐름

| 파일 | 역할 |
|---|---|
| `dynamics.py` | `sub_Dynamics.slx`의 0.1초 단위 단일 UAV 동역학 |
| `guidance.py` | `sub_Guidance.slx` 및 `sub_UAV.slx`의 활성 ACS·PF·VF 선회 유도와 명령 합성 |
| `swarm_env.py` | `Main.slx`, `RL_swarm_env.slx`, `Observations.m`, `Rewards.m` 기반의 5대 동시 갱신, 관측 및 보상 |
| `td3.py` | 기존 PyTorch 벤치마크의 sigmoid Actor와 독립 TD3 에이전트 |
| `train.py` | 환경에서 직접 경험을 수집해 5개 에이전트를 학습하고 가중치를 저장 |

1. `train.py`가 환경을 초기화하고 UAV 5대의 관측값을 받는다.
2. `td3.py`의 Actor가 UAV마다 ACS·PF·VF 가중치 세 개를 계산한다.
3. `guidance.py`가 각 UAV의 상태와 가중치로 뱅크각·속도 명령을 계산한다. ACS는 이웃과의 속도·간격 조정, PF는 가까운 기체로부터의 반발, VF는 목표 궤도 유도를 담당한다.
4. `dynamics.py`가 각 UAV의 위치·기수각·속도·뱅크각을 0.1초 갱신한다.
5. `swarm_env.py`가 갱신된 상태에서 새 관측과 보상을 계산한다. `train.py`는 이 경험을 저장하고, 워밍업이 끝나면 다섯 에이전트를 각각 업데이트한다.

## 3. 원본 재현 범위와 검증

`SwarmEnv.reset()`은 (5, 6) 관측 배열을 반환한다. `step(actions)`에는 (5, 3) 배열을 전달하고, 새 관측, UAV별 보상, 시간 제한 여부를 받는다. 행동 열 순서는 Simulink 배선을 기준으로 ACS, PF, VF이며 하한은 각각 0.1, 0.0, 0.2다. 상태 열 순서는 북쪽 위치, 동쪽 위치, 기수각, 대기속도, 북쪽 속도 성분, 동쪽 속도 성분, 뱅크각이다.

이 프로젝트는 무풍, 고정 위치 초기화, 250 m 반경의 선회 학습 경로를 구현한다. 기본 초기 기수각은 MATLAB의 기존 실행 기록에서 복원한 고정값이다. 다른 초기 상태를 쓰려면 `reset(initial_states=...)`에 (5, 7) 배열을 넘긴다. Simulink의 Unit Delay에 따라 첫 스텝에 가중치 0이 적용되고, 이후에는 이전 스텝에 받은 행동을 적용한다. 또한 원본 `StateReshape`의 배선에 따라 관측·보상 함수의 기수각 항에는 대기속도가 입력된다. 유도와 동역학 계산에는 실제 기수각을 사용한다.

ACS에서 두 기체의 위치가 정확히 같을 때 해당 쌍의 위치 증강항은 0으로 처리한다. 학습 루프는 기본적으로 워밍업 이후 매 스텝 업데이트한다. 체크포인트는 이번 Python 학습의 가중치를 담는다.

기존 Simulink CSV의 2,000스텝 행동을 Python에서 재생해 관측과 보상을 비교했다. 최대 절대 오차는 관측값 `4.72e-7`, 보상 `1.18e-6`이었다. CSV에는 전체 위치 및 자세 상태가 없어 전체 궤적은 별도로 비교해야 한다.

## 4. 강화학습 없이 경로 추종 보기

`simulate.py`는 고정된 ACS·PF·VF 가중치로 5대가 목표 원을 어떻게 추종하는지 보여준다. PyTorch 모델이나 학습된 가중치 파일 없이 실행하며 그래프를 그리려면 Matplotlib가 필요하다.

```bash
cd /home/laaa/td3_swarm_python
python3 simulate.py
```

기본 가중치는 ACS·PF·VF 모두 1이고, 2,000스텝(200초)을 시뮬레이션한다. `runs/trajectory.png`의 왼쪽에 UAV별 경로와 군집 중심, 목표 원이 표시된다. 오른쪽에는 군집 중심의 반경 오차가 표시된다. 0에 가까울수록 목표 원에 가깝고, 음수는 원 안쪽을 뜻한다.

VF 경로 유도만 적용한 결과를 따로 보려면 다음 명령을 실행한다.

```bash
python3 simulate.py --weights 0 0 1 --output runs/vf_only.png
```

움직이는 궤적을 보려면 `python3 simulate.py --animate`를 실행하고 `runs/trajectory.gif`를 연다. 기본 200초 시뮬레이션을 약 10초로 압축해 보여준다.
