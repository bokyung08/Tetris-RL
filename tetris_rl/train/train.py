from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.ppo import get_tetris_policy_kwargs


@dataclass(frozen=True)
class StageGate:
    reward_threshold: float
    length_threshold: float


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """학습 진행률에 따라 learning rate를 선형으로 줄입니다."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


class TetrisCurriculumCallback(BaseCallback):
    """테트리스 PPO용 커리큘럼, 체크포인트, 단계 저장 콜백입니다."""

    def __init__(
        self,
        models_dir: Path,
        gates: dict[int, StageGate] | None = None,
        window_size: int = 30,
        checkpoint_freq: int = 100_000,
        force_stage_after: int = 350_000,
    ) -> None:
        super().__init__(verbose=0)
        self.models_dir = models_dir
        self.gates = gates or {
            0: StageGate(reward_threshold=80.0, length_threshold=50.0),
            1: StageGate(reward_threshold=160.0, length_threshold=70.0),
        }
        self.window_size = window_size
        self.checkpoint_freq = checkpoint_freq
        self.force_stage_after = force_stage_after
        self.recent_rewards: deque[float] = deque(maxlen=window_size)
        self.recent_lengths: deque[float] = deque(maxlen=window_size)
        self.current_stage = 0
        self.stage_start_step = 0
        self.last_checkpoint_step = 0

    def _on_training_start(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        stages = self.training_env.env_method("get_stage")
        self.current_stage = int(stages[0]) if stages else 0
        self.stage_start_step = self.num_timesteps
        print(f"테트리스 전용 PPO 학습 시작: Stage {self.current_stage}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")
            if episode_info:
                self.recent_rewards.append(float(episode_info["r"]))
                self.recent_lengths.append(float(episode_info["l"]))

        self._save_periodic_checkpoint()
        self._advance_stage_if_ready()
        return True

    def _save_periodic_checkpoint(self) -> None:
        if self.checkpoint_freq <= 0:
            return
        if self.num_timesteps - self.last_checkpoint_step < self.checkpoint_freq:
            return

        path = self.models_dir / "tetris_ppo_latest.zip"
        self.model.save(str(path))
        self.last_checkpoint_step = self.num_timesteps
        print(f"중간 체크포인트 저장 완료: {path} ({self.num_timesteps} 스텝)")

    def _advance_stage_if_ready(self) -> None:
        if self.current_stage >= 2:
            return

        enough_episodes = len(self.recent_rewards) >= self.window_size
        stage_steps = self.num_timesteps - self.stage_start_step
        forced = self.force_stage_after > 0 and stage_steps >= self.force_stage_after

        if not enough_episodes and not forced:
            return

        average_reward = float(np.mean(self.recent_rewards)) if self.recent_rewards else -float("inf")
        average_length = float(np.mean(self.recent_lengths)) if self.recent_lengths else 0.0
        gate = self.gates[self.current_stage]
        passed = average_reward >= gate.reward_threshold and average_length >= gate.length_threshold

        if not passed and not forced:
            return

        saved_path = self.save_stage_model(self.current_stage)
        if passed:
            print(
                f"Stage {self.current_stage} 통과: 평균 보상 {average_reward:.2f}, "
                f"평균 생존 {average_length:.2f} 스텝"
            )
        else:
            print(
                f"Stage {self.current_stage} 강제 전환: {stage_steps} 스텝 동안 기준 미달 "
                f"(평균 보상 {average_reward:.2f}, 평균 생존 {average_length:.2f})"
            )
        print(f"단계 모델 저장 완료: {saved_path}")

        self.current_stage += 1
        self.training_env.env_method("set_stage", self.current_stage)
        self.stage_start_step = self.num_timesteps
        self.recent_rewards.clear()
        self.recent_lengths.clear()
        print(f"Stage {self.current_stage}로 전환합니다.")

    def save_stage_model(self, stage: int) -> Path:
        stage_path = self.models_dir / f"stage{stage}.zip"
        named_path = self.models_dir / f"tetris_ppo_stage{stage}.zip"
        self.model.save(str(stage_path))
        self.model.save(str(named_path))
        return stage_path

    def save_final_model(self) -> Path:
        self.save_stage_model(self.current_stage)
        final_path = self.models_dir / "tetris_ppo_final.zip"
        self.model.save(str(final_path))
        print(f"최종 모델 저장 완료: {final_path}")
        return final_path


def make_env(stage: int, max_steps: int, seed: int, rank: int):
    """DummyVecEnv에서 사용할 환경 생성 함수를 반환합니다."""

    def _init() -> Monitor:
        env = TetrisEnv(stage=stage, max_steps=max_steps)
        env.reset(seed=seed + rank)
        return Monitor(env, info_keywords=("stage", "lines_cleared", "max_height"))

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="테트리스 전용 PPO 커리큘럼 학습")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="전체 학습 스텝 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--n-envs", type=int, default=4, help="벡터 환경 수")
    parser.add_argument("--max-steps", type=int, default=5_000, help="에피소드 최대 스텝 수")
    parser.add_argument("--window-size", type=int, default=30, help="단계 전환 평균 계산 에피소드 수")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="중간 모델 저장 주기")
    parser.add_argument("--force-stage-after", type=int, default=350_000, help="기준 미달 시 강제 단계 전환 스텝 수")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "tetris_rl" / "logs", help="TensorBoard 로그 경로")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "tetris_rl" / "models", help="모델 저장 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv(
        [make_env(stage=0, max_steps=args.max_steps, seed=args.seed, rank=rank) for rank in range(args.n_envs)]
    )

    model = PPO(
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
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        verbose=0,
    )

    callback = TetrisCurriculumCallback(
        models_dir=args.model_dir,
        window_size=args.window_size,
        checkpoint_freq=args.checkpoint_freq,
        force_stage_after=args.force_stage_after,
    )

    print("테트리스 전용 PPO 학습을 시작합니다.")
    print("보상은 라인 클리어와 생존을 장려하고, 새 구멍과 위험 높이를 강하게 벌줍니다.")
    print(f"TensorBoard 로그 경로: {args.log_dir}")
    print(f"모델 저장 경로: {args.model_dir}")
    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name="tetris_ppo")
    callback.save_final_model()
    env.close()
    print("학습이 종료되었습니다.")


if __name__ == "__main__":
    main()
