from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_rl.env.tetris_env import TetrisEnv


class CurriculumCallback(BaseCallback):
    """평균 에피소드 보상에 따라 커리큘럼 단계를 자동 전환합니다."""

    def __init__(
        self,
        models_dir: Path,
        thresholds: dict[int, float] | None = None,
        window_size: int = 20,
    ) -> None:
        super().__init__(verbose=0)
        self.models_dir = models_dir
        self.thresholds = thresholds or {0: 200.0, 1: 500.0}
        self.window_size = window_size
        self.recent_rewards: deque[float] = deque(maxlen=window_size)
        self.current_stage = 0

    def _on_training_start(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        stages = self.training_env.env_method("get_stage")
        self.current_stage = int(stages[0]) if stages else 0
        print(f"커리큘럼 학습 시작: Stage {self.current_stage}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")
            if episode_info:
                self.recent_rewards.append(float(episode_info["r"]))

        self._advance_stage_if_ready()
        return True

    def _advance_stage_if_ready(self) -> None:
        if self.current_stage >= 2:
            return
        if len(self.recent_rewards) < self.window_size:
            return

        average_reward = float(np.mean(self.recent_rewards))
        threshold = self.thresholds[self.current_stage]
        if average_reward < threshold:
            return

        saved_path = self._save_stage_model(self.current_stage)
        print(
            f"Stage {self.current_stage} 완료: 최근 {self.window_size}개 에피소드 평균 보상 "
            f"{average_reward:.2f}이 기준 {threshold:.2f} 이상입니다."
        )
        print(f"모델 저장 완료: {saved_path}")

        self.current_stage += 1
        self.training_env.env_method("set_stage", self.current_stage)
        self.recent_rewards.clear()
        print(f"Stage {self.current_stage}로 전환합니다.")

    def save_current_stage_model(self) -> Path:
        saved_path = self._save_stage_model(self.current_stage)
        print(f"최종 모델 저장 완료: {saved_path}")
        return saved_path

    def _save_stage_model(self, stage: int) -> Path:
        path = self.models_dir / f"stage{stage}.zip"
        self.model.save(str(path))
        return path


def make_env(stage: int, max_steps: int, seed: int, rank: int):
    """DummyVecEnv에서 사용할 환경 생성 함수를 반환합니다."""

    def _init() -> Monitor:
        env = TetrisEnv(stage=stage, max_steps=max_steps)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO 기반 테트리스 커리큘럼 학습")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="전체 학습 스텝 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--n-envs", type=int, default=1, help="병렬 환경 수")
    parser.add_argument("--max-steps", type=int, default=5_000, help="에피소드 최대 스텝 수")
    parser.add_argument("--window-size", type=int, default=20, help="단계 전환 평균 보상 계산 에피소드 수")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "tetris_rl" / "logs", help="TensorBoard 로그 경로")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "tetris_rl" / "models", help="모델 저장 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(stage=0, max_steps=args.max_steps, seed=args.seed, rank=rank) for rank in range(args.n_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        verbose=0,
    )

    callback = CurriculumCallback(models_dir=args.model_dir, window_size=args.window_size)

    print("PPO 학습을 시작합니다.")
    print(f"TensorBoard 로그 경로: {args.log_dir}")
    print(f"모델 저장 경로: {args.model_dir}")
    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name="ppo_tetris")
    callback.save_current_stage_model()
    env.close()
    print("학습이 종료되었습니다.")


if __name__ == "__main__":
    main()
