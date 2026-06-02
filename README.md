# PPO 기반 테트리스 강화학습

Gymnasium 커스텀 테트리스 환경에서 Stable-Baselines3 PPO 에이전트를 학습하고, 별도 휴리스틱 기준선과 같은 조건으로 성능을 비교하는 강화학습 프로젝트입니다.

## 프로젝트 목표

- 10x20 테트리스 보드를 즉시 배치형 강화학습 환경으로 구현
- I, O, T, S, Z, J, L 7종 블록과 next block 큐 지원
- PPO 커리큘럼 학습으로 단순 블록 단계에서 전체 블록 단계까지 확장
- 학습된 PPO 모델을 단순 휴리스틱 정책과 비교해 개선율 기록
- 평가 결과를 CSV, JSON, PNG 산출물로 저장해 포트폴리오 지표로 활용

## 기술 스택

- Python 3.10+
- Gymnasium
- Stable-Baselines3 PPO
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
  - 라인 클리어: `+100 x cleared_lines^2`
  - 새 구멍 패널티: `-5 x new_holes`
  - 총 높이 패널티: `-0.5 x aggregate_height`
  - 굴곡도 패널티: `-0.3 x bumpiness`
  - 생존 보너스: `+0.1`
  - 게임오버 패널티: `-50`

명세의 상태 항목을 모두 포함하면 `10+10+7+7+1=35`차원이므로, 항목 누락 없이 35차원 observation으로 구현했습니다.

### PPO 커리큘럼 학습

[tetris_rl/train/train.py](./tetris_rl/train/train.py)

- Stage 0: I, O 블록만 사용
- Stage 1: 전체 블록 사용
- Stage 2: 전체 블록과 next block 정보 사용
- 최근 에피소드 평균 보상이 기준을 넘으면 자동으로 다음 단계로 전환
- 단계별 모델 저장: `tetris_rl/models/stage{n}.zip`
- TensorBoard 로그 저장: `tetris_rl/logs/`

### 휴리스틱 기준선

[tetris_hu/heuristic_policy.py](./tetris_hu/heuristic_policy.py)

휴리스틱은 학습 없이 현재 블록의 40개 배치 후보를 모두 미리 평가합니다. 라인 클리어를 보상하고, 구멍 수, 총 높이, 굴곡도, 게임오버 위험을 감점해 가장 높은 점수의 행동을 선택합니다.

이 기준선은 PPO가 단순한 handcrafted 정책 대비 얼마나 개선됐는지 기록하기 위한 비교 대상으로 사용됩니다.

### 평가 및 비교 리포트

[tetris_rl/eval/evaluate.py](./tetris_rl/eval/evaluate.py)

저장된 PPO 모델과 휴리스틱을 같은 stage, 같은 seed, 같은 에피소드 수로 평가합니다.

저장 산출물:

- `tetris_rl/logs/evaluation_episodes.csv`: 에피소드별 생존 스텝, 라인 수, 보상
- `tetris_rl/logs/performance_comparison.json`: 평균 성능 및 개선율
- `tetris_rl/logs/performance_comparison.csv`: 지표별 PPO vs 휴리스틱 비교
- `tetris_rl/logs/reward_curve.png`: 학습/평가 보상 곡선

개선율 계산:

```text
(PPO 평균 - 휴리스틱 평균) / abs(휴리스틱 평균) x 100
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

PPO 학습:

```powershell
python -m tetris_rl.train.train
```

짧은 테스트 학습:

```powershell
python -m tetris_rl.train.train --total-timesteps 10000
```

PPO와 휴리스틱 비교 평가:

```powershell
python -m tetris_rl.eval.evaluate
```

특정 모델 평가:

```powershell
python -m tetris_rl.eval.evaluate --model tetris_rl/models/stage2.zip
```

휴리스틱 비교 없이 PPO만 평가:

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

평가 실행 후 `performance_comparison.json`과 `performance_comparison.csv`를 통해 PPO가 휴리스틱보다 어떤 지표에서 얼마나 개선됐는지 확인할 수 있습니다.

## 확장 아이디어

- CNN 기반 보드 관측으로 상태 표현 확장
- Hold piece, hard drop, soft drop 등 실제 테트리스 액션 추가
- Optuna 기반 PPO 하이퍼파라미터 탐색
- 휴리스틱 가중치 탐색을 통한 더 강한 기준선 구성
- 평가 결과를 Streamlit 대시보드로 시각화
