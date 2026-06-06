from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.ppo import get_tetris_policy_kwargs
from tetris_rl.train.imitation_utils import collect_imitation_dataset, parse_stages, train_behavior_cloning
from tetris_rl.train.train import linear_schedule


def make_training_env(stage: int, max_steps: int, seed: int) -> DummyVecEnv:
    """MaskablePPO 모델 생성에 사용할 벡터 환경을 만듭니다."""

    def _init() -> Monitor:
        env = TetrisEnv(stage=stage, max_steps=max_steps)
        env.reset(seed=seed)
        return Monitor(env)

    return DummyVecEnv([_init])


def build_maskable_ppo(env: DummyVecEnv, seed: int, log_dir: Path) -> MaskablePPO:
    return MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(3e-4),
        n_steps=1024,
        batch_size=256,
        n_epochs=6,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.6,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=get_tetris_policy_kwargs(),
        tensorboard_log=str(log_dir),
        seed=seed,
        verbose=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="휴리스틱 정책 기반 MaskablePPO imitation pretraining")
    parser.add_argument("--stages", type=str, default="0,1,2", help="데이터를 수집할 stage 목록. 예: 0 또는 0,1,2")
    parser.add_argument("--samples-per-stage", type=int, default=5_000, help="stage별 수집 샘플 수")
    parser.add_argument("--max-steps", type=int, default=500, help="데이터 수집 에피소드 최대 스텝 수")
    parser.add_argument("--epochs", type=int, default=10, help="지도학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=256, help="지도학습 배치 크기")
    parser.add_argument("--entropy-coef", type=float, default=0.001, help="초기 policy 탐색성을 유지하기 위한 entropy 계수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / "tetris_rl" / "models" / "tetris_maskable_ppo_imitation.zip",
        help="사전학습 모델 저장 경로",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "tetris_rl" / "logs",
        help="TensorBoard 로그 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = parse_stages(args.stages)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print("휴리스틱 imitation 데이터 수집을 시작합니다.")
    dataset = collect_imitation_dataset(
        stages=stages,
        samples_per_stage=args.samples_per_stage,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(f"전체 데이터 수집 완료: {len(dataset.actions)}개")

    env = make_training_env(stage=stages[0], max_steps=args.max_steps, seed=args.seed)
    model = build_maskable_ppo(env=env, seed=args.seed, log_dir=args.log_dir)
    train_behavior_cloning(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
    )

    model.save(str(args.output_model))
    env.close()
    print(f"사전학습 모델 저장 완료: {args.output_model}")
    print("이 모델로 PPO fine-tuning을 시작하려면 다음 명령을 사용하세요.")
    print(f"python -m tetris_rl.train.train --pretrained-model {args.output_model}")


if __name__ == "__main__":
    main()
