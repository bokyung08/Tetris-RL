# MaskablePPO 기반 테트리스 강화학습

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Custom%20Env-008080)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-EE4C2C)
![SB3-Contrib](https://img.shields.io/badge/SB3--Contrib-MaskablePPO-5F4B8B)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom%20Extractor-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Logging-FF6F00?logo=tensorflow&logoColor=white)
![Benchmark](https://img.shields.io/badge/Benchmark-Heuristic%20vs%20MaskablePPO-6A5ACD)

Gymnasium 커스텀 테트리스 환경에서 SB3-Contrib MaskablePPO 에이전트를 학습하고, 별도 휴리스틱 기준선과 같은 조건으로 성능을 비교하는 강화학습 프로젝트입니다.

## 프로젝트 목표

- 10x20 테트리스 보드를 즉시 배치형 강화학습 환경으로 구현
- I, O, T, S, Z, J, L 7종 블록과 next block 큐 지원
- MaskablePPO 커리큘럼 학습으로 단순 블록 단계에서 전체 블록 단계까지 확장
- 불가능한 열/회전 행동을 action mask로 제거해 학습 신호 개선
- 학습된 MaskablePPO 모델을 단순 휴리스틱 정책과 비교해 개선율 기록
- 평가 결과를 CSV, JSON, PNG 산출물로 저장해 포트폴리오 지표로 활용

## 기술 스택

- Python 3.10+
- Gymnasium
- Stable-Baselines3
- SB3-Contrib MaskablePPO
- PyTorch
- NumPy
- Matplotlib
- TensorBoard

## 주요 구현

### 강화학습 환경

[tetris_rl/env/tetris_env.py](./tetris_rl/env/tetris_env.py)

- 보드 크기: 10x20
- 행동 공간: `열 10개 x 회전 4개 = Discrete(40)`
- 블록: I, O, T, S, Z, J, L
- 상태 벡터: 열 높이, 열 구멍 수, 현재 블록 원핫, 다음 블록 원핫, 커리큘럼 단계
- 보상:
  - 라인 클리어: Stage 1 기준 1줄 `+40`, 2줄 `+120`, 3줄 `+300`, 4줄 `+800`
  - 생존 보너스: Stage 0/1 기준 `+2.0`
  - 안전한 배치 보너스: 새 구멍이 없으면 보너스 지급
  - 새 구멍 및 구멍 증가 패널티: 구멍 증가량 중심 감점
  - 높이/굴곡도 패널티: 전체 값이 아니라 이번 행동으로 악화된 변화량 중심 감점
  - 위험 높이 패널티: 최대 높이가 일정 기준을 넘으면 제곱 패널티
  - 게임오버 패널티: Stage 0/1 기준 `-80`

Stage별 보상 스케줄:

- Stage 0: 구멍/높이/굴곡도 패널티를 약하게 적용해 생존 전략을 먼저 학습
- Stage 1: 전체 블록으로 확장하면서 패널티를 중간 강도로 적용
- Stage 2: next block 정보까지 사용하며 구멍과 위험 높이 패널티를 강화

명세의 상태 항목을 모두 포함하면 `10+10+7+7+1=35`차원이므로, 항목 누락 없이 35차원 observation으로 구현했습니다.

### 테트리스 전용 MaskablePPO

[tetris_rl/ppo/tetris_policy.py](./tetris_rl/ppo/tetris_policy.py)

일반 MLP에 35차원 벡터를 그대로 넣지 않고, 상태를 테트리스 구조에 맞게 나눠 처리합니다.

- 열 높이 10개 전용 encoder
- 열 구멍 수 10개 전용 encoder
- 현재/다음 블록 one-hot encoder
- 총 높이, 총 구멍, 굴곡도, 최대 높이, 높이 범위, stage 등 파생 지표 encoder
- MaskablePPO actor와 critic은 이 feature extractor의 결과를 사용

기존 보상은 매 스텝 총 높이를 계속 벌줘 오래 살아남을수록 누적 손해가 커질 수 있었습니다. 현재 버전은 행동 전후 변화량을 중심으로 보상을 계산하고, 초반 stage에서는 벌점을 완화해 MaskablePPO가 생존 전략을 먼저 잡도록 수정했습니다.

Action Masking:

- 현재 블록의 회전 상태별 폭을 계산합니다.
- 보드 오른쪽 밖으로 나가는 열 선택은 `False`로 마스킹합니다.
- MaskablePPO는 마스킹된 행동을 선택하지 않으므로, 보정된 열 때문에 생기는 모호한 학습 신호를 줄입니다.

### MaskablePPO 커리큘럼 학습

[tetris_rl/train/train.py](./tetris_rl/train/train.py)

- Stage 0: I, O 블록만 사용
- Stage 1: 전체 블록 사용
- Stage 2: 전체 블록과 next block 정보 사용
- 최근 에피소드 평균 보상과 평균 생존 스텝이 기준을 넘으면 자동으로 다음 단계로 전환
- 기준에 오래 못 미치면 지정된 스텝 이후 다음 stage로 강제 전환해 학습 정체를 방지
- 단계별 모델 저장: `tetris_rl/models/stage{n}.zip`
- 전용 모델 저장: `tetris_rl/models/tetris_maskable_ppo_stage{n}.zip`, `tetris_rl/models/tetris_maskable_ppo_final.zip`
- 중간 체크포인트 저장: `tetris_rl/models/tetris_maskable_ppo_latest.zip`
- TensorBoard 로그 저장: `tetris_rl/logs/`

### 휴리스틱 기준선

[tetris_hu/heuristic_policy.py](./tetris_hu/heuristic_policy.py)

휴리스틱은 학습 없이 현재 블록의 40개 배치 후보를 모두 미리 평가합니다. 라인 클리어를 보상하고, 구멍 수, 총 높이, 굴곡도, 게임오버 위험을 감점해 가장 높은 점수의 행동을 선택합니다.

이 기준선은 MaskablePPO가 단순한 handcrafted 정책 대비 얼마나 개선됐는지 기록하기 위한 비교 대상으로 사용됩니다.

### 평가 및 비교 리포트

[tetris_rl/eval/evaluate.py](./tetris_rl/eval/evaluate.py)

저장된 MaskablePPO 모델과 휴리스틱을 같은 stage, 같은 seed, 같은 에피소드 수로 평가합니다.

저장 산출물:

- `tetris_rl/logs/evaluation_episodes.csv`: 에피소드별 생존 스텝, 라인 수, 보상
- `tetris_rl/logs/performance_comparison.json`: 평균 성능 및 개선율
- `tetris_rl/logs/performance_comparison.csv`: 지표별 MaskablePPO vs 휴리스틱 비교
- `tetris_rl/logs/reward_curve.png`: 학습/평가 보상 곡선

개선율 계산:

```text
(MaskablePPO 평균 - 휴리스틱 평균) / abs(휴리스틱 평균) x 100
```

## 프로젝트 구조

```text
tetris/
├── tetris_hu/
│   ├── __init__.py
│   └── heuristic_policy.py
├── tetris_rl/
│   ├── env/
│   │   └── tetris_env.py
│   ├── train/
│   │   └── train.py
│   ├── eval/
│   │   └── evaluate.py
│   ├── ppo/
│   │   └── tetris_policy.py
│   ├── logs/
│   └── models/
├── requirements.txt
└── README.md
```

## 실행 방법

의존성 설치:

```powershell
pip install -r requirements.txt
```

MaskablePPO 학습:

```powershell
python -m tetris_rl.train.train
```

짧은 테스트 학습:

```powershell
python -m tetris_rl.train.train --total-timesteps 10000
```

MaskablePPO와 휴리스틱 비교 평가:

```powershell
python -m tetris_rl.eval.evaluate
```

특정 모델 평가:

```powershell
python -m tetris_rl.eval.evaluate --model tetris_rl/models/stage2.zip
```

휴리스틱 비교 없이 MaskablePPO만 평가:

```powershell
python -m tetris_rl.eval.evaluate --skip-heuristic
```

TensorBoard 확인:

```powershell
tensorboard --logdir tetris_rl/logs
```

## 포트폴리오 관점의 성과 지표

이 프로젝트는 모델의 단순 학습 여부만 보지 않고, 아래 지표를 기준선 대비 개선율로 기록합니다.

- 평균 생존 스텝
- 평균 라인 클리어 수
- 평균 보상

평가 실행 후 `performance_comparison.json`과 `performance_comparison.csv`를 통해 MaskablePPO가 휴리스틱보다 어떤 지표에서 얼마나 개선됐는지 확인할 수 있습니다.

## 확장 아이디어

- CNN 기반 보드 관측으로 상태 표현 확장
- Hold piece, hard drop, soft drop 등 실제 테트리스 액션 추가
- Optuna 기반 MaskablePPO 하이퍼파라미터 탐색
- 휴리스틱 가중치 탐색을 통한 더 강한 기준선 구성
- 평가 결과를 Streamlit 대시보드로 시각화
