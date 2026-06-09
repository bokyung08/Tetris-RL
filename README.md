# MaskablePPO 기반 테트리스 강화학습

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Custom%20Env-008080)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-EE4C2C)
![SB3-Contrib](https://img.shields.io/badge/SB3--Contrib-MaskablePPO-5F4B8B)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom%20Extractor-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Pygame](https://img.shields.io/badge/Pygame-Simulation-2F8F46)
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
- Pygame
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
  - Stage 2 테트리스 지향 보상: 1줄 `+35`, 2줄 `+180`, 3줄 `+700`, 4줄 `+2600`
  - 생존 보너스: Stage 0/1 기준 `+2.0`
  - 안전한 배치 보너스: 새 구멍이 없으면 보너스 지급
  - 새 구멍 및 구멍 증가 패널티: 구멍 증가량 중심 감점
  - 높이/굴곡도 패널티: 전체 값이 아니라 이번 행동으로 악화된 변화량 중심 감점
  - 위험 높이 패널티: 최대 높이가 일정 기준을 넘으면 제곱 패널티
  - 게임오버 패널티: Stage 0/1 기준 `-80`

Stage별 보상 스케줄:

- Stage 0: 구멍/높이/굴곡도 패널티를 약하게 적용해 생존 전략을 먼저 학습
- Stage 1: 전체 블록으로 확장하면서 패널티를 중간 강도로 적용
- Stage 2: next block 정보까지 사용하며 구멍과 위험 높이 패널티를 강화하고, 4줄 동시 클리어를 더 크게 보상

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
- 기준 미달 상태의 강제 stage 전환은 기본 비활성화되어 있으며, `--force-stage-after`로만 켭니다.
- 단계별 모델 저장: `tetris_rl/models/stage{n}.zip`
- 전용 모델 저장: `tetris_rl/models/tetris_maskable_ppo_stage{n}.zip`, `tetris_rl/models/tetris_maskable_ppo_final.zip`
- 중간 체크포인트 저장: `tetris_rl/models/tetris_maskable_ppo_latest.zip`
- Stage별 best 모델 저장: `tetris_rl/models/tetris_maskable_ppo_best_stage{n}.zip`
- 전체 stage 평균 기준 best 모델 저장: `tetris_rl/models/tetris_maskable_ppo_best.zip`
- TensorBoard 로그 저장: `tetris_rl/logs/`

### 휴리스틱 Imitation Pretraining

[tetris_rl/train/pretrain_imitation.py](./tetris_rl/train/pretrain_imitation.py)

랜덤 정책에서 PPO만 시작하면 테트리스는 보상 신호가 늦게 오고 희소해서 배치 전략을 잡기 어렵습니다. 이를 보완하기 위해 휴리스틱을 교사 정책으로 사용합니다.

- 휴리스틱 정책으로 `(상태, action mask, 행동)` 데이터셋 수집
- `--stage-samples`로 Stage 1/2 데이터를 더 강하게 줄 수 있음
- MaskablePPO actor를 cross entropy로 지도학습
- 사전학습 종료 시 전체 및 stage별 선생님 행동 일치율 출력
- 사전학습 모델 저장: `tetris_rl/models/tetris_maskable_ppo_imitation.zip`
- 이후 PPO fine-tuning은 `--pretrained-model` 옵션으로 시작

### 휴리스틱 기준선

[tetris_hu/heuristic_policy.py](./tetris_hu/heuristic_policy.py)

휴리스틱은 학습 없이 현재 블록의 40개 배치 후보를 모두 미리 평가합니다. 라인 클리어를 보상하고, 구멍 수, 총 높이, 굴곡도, 게임오버 위험을 감점해 가장 높은 점수의 행동을 선택합니다.

이 기준선은 MaskablePPO가 단순한 handcrafted 정책 대비 얼마나 개선됐는지 기록하기 위한 비교 대상으로 사용됩니다.

### 평가 및 비교 리포트

[tetris_rl/eval/evaluate.py](./tetris_rl/eval/evaluate.py)

저장된 MaskablePPO 모델과 휴리스틱을 같은 stage, 같은 seed, 같은 에피소드 수로 평가합니다.

저장 산출물:

- `tetris_rl/logs/evaluation_episodes.csv`: 에피소드별 생존 스텝, 라인 수, 1/2/3/4줄 동시 클리어 횟수, 보상
- `tetris_rl/logs/performance_comparison.json`: 평균 성능 및 개선율
- `tetris_rl/logs/performance_comparison.csv`: 생존, 총 라인, 보상, 동시 클리어 지표별 MaskablePPO vs 휴리스틱 비교
- `tetris_rl/logs/reward_curve.png`: 학습/평가 보상 곡선

개선율 계산:

```text
(MaskablePPO 평균 - 휴리스틱 평균) / abs(휴리스틱 평균) x 100
```

### Pygame 시뮬레이션

[tetris_rl/eval/watch_pygame.py](./tetris_rl/eval/watch_pygame.py)

Gymnasium 환경을 그대로 step하면서 학습 모델 또는 휴리스틱 정책의 배치 결과를 pygame 창으로 확인합니다.

- 기본 모델 탐색은 `tetris_maskable_ppo_best.zip`을 우선 사용
- 모델 정책과 휴리스틱 정책 모두 지원
- `Space`: 일시정지/재개
- `N`: 한 스텝 진행
- `R`: 에피소드 재시작
- `Q` 또는 `Esc`: 종료

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
│   │   ├── train.py
│   │   └── pretrain_imitation.py
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

휴리스틱 imitation 사전학습:

```powershell
python -m tetris_rl.train.pretrain_imitation
```

사전학습 모델에서 MaskablePPO fine-tuning:

```powershell
python -m tetris_rl.train.train --pretrained-model tetris_rl/models/tetris_maskable_ppo_imitation.zip
```

사전학습 모델을 사용할 때는 기본 fine-tuning 설정이 더 보수적으로 적용됩니다.

- learning rate: `2e-5`
- entropy coefficient: `0.0`
- clip range: `0.03`
- PPO epochs: `1`
- target KL: `0.005`
- stage 전환 전 최소 학습 스텝: `150000`
- 강제 stage 전환: 기본 비활성화

fine-tuning 중 policy가 휴리스틱 teacher에서 너무 멀어지면 성능이 무너질 수 있어, BC 정규화를 함께 사용할 수 있습니다.

- PPO rollout 사이에 휴리스틱 행동 지도학습을 반복 적용
- 전체 및 Stage 0/1/2별 선생님 행동 일치율을 TensorBoard에 기록
- 평가 성능이 가장 좋은 모델을 전체 평균 best와 stage별 best로 분리 저장
- PPO와 DQN을 한 루프에 섞기보다, 먼저 BC-regularized PPO로 teacher policy 붕괴를 막는 설계

```powershell
python -m tetris_rl.train.train --pretrained-model tetris_rl/models/tetris_maskable_ppo_imitation.zip --bc-regularize
```

Stage 1/2 선생님을 강하게 주는 권장 루트:

```powershell
python -m tetris_rl.train.pretrain_imitation --stage-samples "0:5000,1:30000,2:30000" --epochs 30 --entropy-coef 0.0 --output-model tetris_rl/models/tetris_maskable_ppo_imitation_stage12_strong.zip --log-dir tetris_rl/logs/imitation_stage12_strong
```

```powershell
python -m tetris_rl.train.train --start-stage 1 --pretrained-model tetris_rl/models/tetris_maskable_ppo_imitation_stage12_strong.zip --bc-regularize --bc-stage-samples "0:1000,1:20000,2:20000" --total-timesteps 700000 --force-stage-after 0 --model-dir tetris_rl/models/bc_stage12_strong --log-dir tetris_rl/logs/bc_stage12_strong
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

Pygame으로 모델 시뮬레이션 보기:

```powershell
python -m tetris_rl.eval.watch_pygame --model tetris_rl/models/tetris_maskable_ppo_imitation.zip --stage 0
```

Pygame으로 best 모델 보기:

```powershell
python -m tetris_rl.eval.watch_pygame --stage 0
```

Pygame으로 휴리스틱 기준선 보기:

```powershell
python -m tetris_rl.eval.watch_pygame --policy heuristic --stage 0
```

TensorBoard 확인:

```powershell
tensorboard --logdir tetris_rl/logs
```

## 성과 지표

이 프로젝트는 모델의 단순 학습 여부만 보지 않고, 아래 지표를 기준선 대비 개선율로 기록

- 평균 생존 스텝
- 평균 라인 클리어 수
- 평균 1/2/3/4줄 동시 클리어 횟수
- 평균 보상

평가 실행 후 `performance_comparison.json`과 `performance_comparison.csv`를 통해 MaskablePPO가 휴리스틱보다 어떤 지표에서 얼마나 개선됐는지 확인할 수 있습니다.

## 보존 모델과 최근 실험 결과

현재 보존된 Stage 2 개선 모델:

```text
tetris_rl/models/stage2_finetune_v2/tetris_maskable_ppo_final_0608_v2.zip
tetris_rl/models/stage2_tetris_bonus_v1/tetris_maskable_ppo_best_stage2_0609_tetris_bonus.zip
tetris_rl/models/stage2_tetris_bonus_v2/tetris_maskable_ppo_final_0609_tetris_bonus_v2.zip
```

`tetris_maskable_ppo_final_0609_tetris_bonus_v2.zip`는 Stage 2 테트리스 지향 보상 기준에서 현재 가장 좋은 보존 모델입니다.

평가 조건:

- Stage 2
- 50 에피소드
- 에피소드 최대 500 스텝
- deterministic policy
- 동일 seed 구간
- Stage 2 보상: 1줄 `+35`, 2줄 `+180`, 3줄 `+700`, 4줄 `+2600`

| 모델 | 평균 보상 | 중앙값 보상 | 평균 생존 | 평균 라인 | 1줄 | 2줄 | 3줄 | 4줄 | 4줄 총합 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bonus_v1_keep` | 1473.38 | 1312.83 | 164.30 | 49.70 | 42.58 | 3.18 | 0.20 | 0.04 | 2 |
| `bonus_v2_best_stage2` | 1473.38 | 1312.83 | 164.30 | 49.70 | 42.58 | 3.18 | 0.20 | 0.04 | 2 |
| `bonus_v2_final` | 1816.02 | 1738.88 | 180.26 | 55.86 | 47.32 | 3.70 | 0.30 | 0.06 | 3 |

분석:

- `bonus_v2_final`이 평균 보상, 중앙값 보상, 생존 스텝, 총 라인 수에서 모두 가장 높았습니다.
- 4줄 동시 클리어는 50 에피소드 기준 `2회 -> 3회`로 늘었습니다. 아직 큰 폭은 아니지만, 보상 방향은 4줄 클리어를 더 선호하는 쪽으로 움직였습니다.
- `bonus_v2_best_stage2`는 TensorBoard best 파일이지만, 실제 50 에피소드 직접 평가에서는 초기 모델과 같은 성능이었습니다. Stage 2는 분산이 커서 best 파일만 믿기보다 `best_stage2.zip`과 `final.zip`을 같은 seed로 재평가하는 과정이 필요합니다.
- 1줄 클리어도 같이 증가했기 때문에, 다음 실험에서는 4줄 클리어를 더 늘리려면 1줄 보상을 추가로 낮추거나 휴리스틱 teacher 자체를 4줄 클리어 지향으로 조정하는 것이 더 직접적입니다.

테트리스 보너스 v2 재현 명령:

```powershell
python -m tetris_rl.train.train `
  --start-stage 2 `
  --pretrained-model tetris_rl/models/stage2_tetris_bonus_v1/tetris_maskable_ppo_best_stage2_0609_tetris_bonus.zip `
  --bc-regularize `
  --bc-stages 2 `
  --bc-stage-samples "2:60000" `
  --bc-max-steps 800 `
  --total-timesteps 350000 `
  --force-stage-after 0 `
  --eval-stages 2 `
  --eval-freq 25000 `
  --eval-episodes 50 `
  --eval-max-steps 500 `
  --pretrained-learning-rate 5e-6 `
  --pretrained-clip-range 0.012 `
  --pretrained-target-kl 0.002 `
  --pretrained-ent-coef 0.0 `
  --pretrained-n-epochs 1 `
  --pretrained-batch-size 1024 `
  --bc-update-freq 8192 `
  --bc-epochs-per-update 2 `
  --bc-entropy-coef 0.0 `
  --model-dir tetris_rl/models/stage2_tetris_bonus_v2 `
  --log-dir tetris_rl/logs/stage2_tetris_bonus_v2
```

학습 후 4줄 클리어가 실제로 늘었는지 확인하는 평가 명령:

```powershell
python -m tetris_rl.eval.evaluate --model tetris_rl/models/stage2_tetris_bonus_v2/tetris_maskable_ppo_final_0609_tetris_bonus_v2.zip --stage 2 --episodes 100 --max-steps 500 --skip-heuristic --log-dir tetris_rl/logs/stage2_tetris_bonus_v2_eval
```

## 확장 아이디어

- CNN 기반 보드 관측으로 상태 표현 확장
- Hold piece, hard drop, soft drop 등 실제 테트리스 액션 추가
- Optuna 기반 MaskablePPO 하이퍼파라미터 탐색
- 휴리스틱 가중치 탐색을 통한 더 강한 기준선 구성
- 평가 결과를 Streamlit 대시보드로 시각화
