from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.ppo import get_tetris_policy_kwargs
from tetris_rl.train.imitation_utils import ImitationDataset, collect_imitation_dataset, parse_stages, train_behavior_cloning


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
        min_stage_steps: int = 150_000,
    ) -> None:
        super().__init__(verbose=0)
        self.models_dir = models_dir
        self.gates = gates or {
            0: StageGate(reward_threshold=40.0, length_threshold=45.0),
            1: StageGate(reward_threshold=80.0, length_threshold=60.0),
        }
        self.window_size = window_size
        self.checkpoint_freq = checkpoint_freq
        self.force_stage_after = force_stage_after
        self.min_stage_steps = min_stage_steps
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
        print(f"테트리스 전용 MaskablePPO 학습 시작: Stage {self.current_stage}")

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

        path = self.models_dir / "tetris_maskable_ppo_latest.zip"
        self.model.save(str(path))
        self.last_checkpoint_step = self.num_timesteps
        print(f"중간 체크포인트 저장 완료: {path} ({self.num_timesteps} 스텝)")

    def _advance_stage_if_ready(self) -> None:
        if self.current_stage >= 2:
            return

        enough_episodes = len(self.recent_rewards) >= self.window_size
        stage_steps = self.num_timesteps - self.stage_start_step
        forced = self.force_stage_after > 0 and stage_steps >= self.force_stage_after

        if stage_steps < self.min_stage_steps:
            return

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
        named_path = self.models_dir / f"tetris_maskable_ppo_stage{stage}.zip"
        self.model.save(str(stage_path))
        self.model.save(str(named_path))
        return stage_path

    def save_final_model(self) -> Path:
        self.save_stage_model(self.current_stage)
        final_path = self.models_dir / "tetris_maskable_ppo_final.zip"
        self.model.save(str(final_path))
        print(f"최종 모델 저장 완료: {final_path}")
        return final_path


def evaluate_model_on_env(
    model: MaskablePPO,
    *,
    stage: int,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    """현재 모델을 deterministic policy로 짧게 평가합니다."""
    env = TetrisEnv(stage=stage, max_steps=max_steps)
    rewards: list[float] = []
    steps: list[int] = []
    lines: list[int] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        info = {"steps": 0, "lines_cleared": 0}

        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_reward += float(reward)
            done = terminated or truncated

        rewards.append(total_reward)
        steps.append(int(info["steps"]))
        lines.append(int(info["lines_cleared"]))

    env.close()
    return {
        "reward": float(np.mean(rewards)),
        "steps": float(np.mean(steps)),
        "lines": float(np.mean(lines)),
    }


class BestModelEvalCallback(BaseCallback):
    """평가 성능이 가장 좋은 모델을 별도 파일로 보존합니다."""

    def __init__(
        self,
        models_dir: Path,
        eval_freq: int,
        eval_episodes: int,
        eval_max_steps: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=0)
        self.models_dir = models_dir
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_max_steps = eval_max_steps
        self.seed = seed
        self.best_score = -float("inf")
        self.last_eval_step = 0

    def _on_training_start(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._evaluate_and_save(reason="초기 모델")

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate_and_save(reason=f"{self.num_timesteps} 스텝")
            self.last_eval_step = self.num_timesteps
        return True

    def _evaluate_and_save(self, reason: str) -> None:
        stages = self.training_env.env_method("get_stage")
        stage = int(stages[0]) if stages else 0
        metrics = evaluate_model_on_env(
            self.model,
            stage=stage,
            episodes=self.eval_episodes,
            max_steps=self.eval_max_steps,
            seed=self.seed,
        )
        score = metrics["reward"]
        print(
            f"Best 평가({reason}, Stage {stage}): 평균 보상 {metrics['reward']:.2f}, "
            f"평균 생존 {metrics['steps']:.2f}, 평균 라인 {metrics['lines']:.2f}"
        )

        if score > self.best_score:
            self.best_score = score
            best_path = self.models_dir / "tetris_maskable_ppo_best.zip"
            stage_best_path = self.models_dir / f"tetris_maskable_ppo_best_stage{stage}.zip"
            self.model.save(str(best_path))
            self.model.save(str(stage_best_path))
            print(f"최고 성능 모델 저장 완료: {best_path}")


class BehaviorCloningRegularizationCallback(BaseCallback):
    """PPO 업데이트 사이에 휴리스틱 행동 지도학습을 반복해 policy 붕괴를 막습니다."""

    def __init__(
        self,
        dataset: ImitationDataset,
        update_freq: int,
        epochs_per_update: int,
        batch_size: int,
        entropy_coef: float,
    ) -> None:
        super().__init__(verbose=0)
        self.dataset = dataset
        self.update_freq = update_freq
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.last_update_step = 0

    def _on_training_start(self) -> None:
        print(f"BC 정규화 데이터셋 크기: {len(self.dataset.actions)}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.update_freq <= 0:
            return
        if self.num_timesteps - self.last_update_step < self.update_freq:
            return

        print(f"BC 정규화 업데이트 시작: {self.num_timesteps} 스텝")
        loss, accuracy = train_behavior_cloning(
            model=self.model,
            dataset=self.dataset,
            epochs=self.epochs_per_update,
            batch_size=self.batch_size,
            entropy_coef=self.entropy_coef,
        )
        self.last_update_step = self.num_timesteps
        self.logger.record("bc/loss", loss)
        self.logger.record("bc/action_accuracy", accuracy)


def make_env(stage: int, max_steps: int, seed: int, rank: int):
    """DummyVecEnv에서 사용할 환경 생성 함수를 반환합니다."""

    def _init() -> Monitor:
        env = TetrisEnv(stage=stage, max_steps=max_steps)
        env.reset(seed=seed + rank)
        return Monitor(env, info_keywords=("stage", "lines_cleared", "max_height"))

    return _init


def apply_conservative_finetuning_config(model: MaskablePPO, args: argparse.Namespace) -> None:
    """사전학습 policy가 PPO 업데이트로 급격히 망가지지 않도록 설정을 낮춥니다."""
    model.learning_rate = linear_schedule(args.pretrained_learning_rate)
    model.lr_schedule = linear_schedule(args.pretrained_learning_rate)
    model.clip_range = linear_schedule(args.pretrained_clip_range)
    model.ent_coef = args.pretrained_ent_coef
    model.target_kl = args.pretrained_target_kl
    model.n_epochs = args.pretrained_n_epochs
    model.batch_size = args.pretrained_batch_size
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = args.pretrained_learning_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="테트리스 전용 PPO 커리큘럼 학습")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="전체 학습 스텝 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--n-envs", type=int, default=4, help="벡터 환경 수")
    parser.add_argument("--max-steps", type=int, default=5_000, help="에피소드 최대 스텝 수")
    parser.add_argument("--window-size", type=int, default=30, help="단계 전환 평균 계산 에피소드 수")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="중간 모델 저장 주기")
    parser.add_argument("--force-stage-after", type=int, default=350_000, help="기준 미달 시 강제 단계 전환 스텝 수")
    parser.add_argument("--min-stage-steps", type=int, default=150_000, help="단계 전환 전 최소 학습 스텝 수")
    parser.add_argument("--pretrained-model", type=Path, default=None, help="휴리스틱 imitation 사전학습 모델 경로")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="처음부터 학습할 때 learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="처음부터 학습할 때 entropy 계수")
    parser.add_argument("--clip-range", type=float, default=0.15, help="처음부터 학습할 때 PPO clip range")
    parser.add_argument("--n-epochs", type=int, default=6, help="처음부터 학습할 때 PPO epoch 수")
    parser.add_argument("--target-kl", type=float, default=0.03, help="처음부터 학습할 때 target KL")
    parser.add_argument("--pretrained-learning-rate", type=float, default=5e-5, help="사전학습 모델 fine-tuning learning rate")
    parser.add_argument("--pretrained-ent-coef", type=float, default=0.001, help="사전학습 모델 fine-tuning entropy 계수")
    parser.add_argument("--pretrained-clip-range", type=float, default=0.05, help="사전학습 모델 fine-tuning PPO clip range")
    parser.add_argument("--pretrained-n-epochs", type=int, default=2, help="사전학습 모델 fine-tuning PPO epoch 수")
    parser.add_argument("--pretrained-target-kl", type=float, default=0.01, help="사전학습 모델 fine-tuning target KL")
    parser.add_argument("--pretrained-batch-size", type=int, default=512, help="사전학습 모델 fine-tuning 배치 크기")
    parser.add_argument("--bc-regularize", action="store_true", help="PPO fine-tuning 중 휴리스틱 BC 정규화를 반복 적용")
    parser.add_argument("--bc-stages", type=str, default="0,1,2", help="BC 정규화 데이터 수집 stage 목록")
    parser.add_argument("--bc-samples-per-stage", type=int, default=2_000, help="BC 정규화 stage별 샘플 수")
    parser.add_argument("--bc-max-steps", type=int, default=500, help="BC 데이터 수집 에피소드 최대 스텝")
    parser.add_argument("--bc-update-freq", type=int, default=16_384, help="BC 정규화 업데이트 주기")
    parser.add_argument("--bc-epochs-per-update", type=int, default=1, help="BC 정규화 1회당 epoch 수")
    parser.add_argument("--bc-batch-size", type=int, default=512, help="BC 정규화 배치 크기")
    parser.add_argument("--bc-entropy-coef", type=float, default=0.0005, help="BC 정규화 entropy 계수")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="best 모델 평가 주기")
    parser.add_argument("--eval-episodes", type=int, default=10, help="best 모델 평가 에피소드 수")
    parser.add_argument("--eval-max-steps", type=int, default=500, help="best 모델 평가 에피소드 최대 스텝")
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

    if args.pretrained_model:
        model = MaskablePPO.load(str(args.pretrained_model), env=env, tensorboard_log=str(args.log_dir))
        apply_conservative_finetuning_config(model, args)
        print(f"사전학습 모델을 불러왔습니다: {args.pretrained_model}")
        print(
            "보수적 fine-tuning 설정을 적용합니다: "
            f"lr={args.pretrained_learning_rate}, ent_coef={args.pretrained_ent_coef}, "
            f"clip={args.pretrained_clip_range}, n_epochs={args.pretrained_n_epochs}, "
            f"target_kl={args.pretrained_target_kl}"
        )
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=1024,
            batch_size=256,
            n_epochs=args.n_epochs,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.6,
            max_grad_norm=0.5,
            target_kl=args.target_kl,
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
        min_stage_steps=args.min_stage_steps,
    )
    callbacks: list[BaseCallback] = [
        callback,
        BestModelEvalCallback(
            models_dir=args.model_dir,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps,
            seed=args.seed + 10_000,
        ),
    ]

    if args.bc_regularize:
        print("BC 정규화 데이터셋을 수집합니다.")
        bc_dataset = collect_imitation_dataset(
            stages=parse_stages(args.bc_stages),
            samples_per_stage=args.bc_samples_per_stage,
            max_steps=args.bc_max_steps,
            seed=args.seed + 20_000,
        )
        callbacks.append(
            BehaviorCloningRegularizationCallback(
                dataset=bc_dataset,
                update_freq=args.bc_update_freq,
                epochs_per_update=args.bc_epochs_per_update,
                batch_size=args.bc_batch_size,
                entropy_coef=args.bc_entropy_coef,
            )
        )

    print("테트리스 전용 MaskablePPO 학습을 시작합니다.")
    print("보상은 라인 클리어와 생존을 장려하고, 새 구멍과 위험 높이를 강하게 벌줍니다.")
    print("Action Masking으로 현재 블록이 보드 밖으로 나가는 행동은 선택하지 않습니다.")
    print(f"TensorBoard 로그 경로: {args.log_dir}")
    print(f"모델 저장 경로: {args.model_dir}")
    model.learn(total_timesteps=args.total_timesteps, callback=CallbackList(callbacks), tb_log_name="tetris_maskable_ppo")
    callback.save_final_model()
    env.close()
    print("학습이 종료되었습니다.")


if __name__ == "__main__":
    main()
