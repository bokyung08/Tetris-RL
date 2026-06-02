from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.ppo import TetrisFeatureExtractor
from tetris_hu import choose_heuristic_action


_TETRIS_FEATURE_EXTRACTOR = TetrisFeatureExtractor


ActionPolicy = Callable[[np.ndarray, TetrisEnv], int]


@dataclass
class EvaluationResult:
    policy_name: str
    survival_steps: list[int]
    line_counts: list[int]
    rewards: list[float]

    def summary(self) -> dict[str, float]:
        return {
            "average_survival_steps": float(np.mean(self.survival_steps)),
            "average_lines_cleared": float(np.mean(self.line_counts)),
            "average_reward": float(np.mean(self.rewards)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습된 PPO 테트리스 모델 평가 및 휴리스틱 기준선 비교")
    parser.add_argument("--model", type=Path, default=None, help="평가할 모델 파일 경로")
    parser.add_argument("--episodes", type=int, default=100, help="평가 에피소드 수")
    parser.add_argument("--stage", type=int, default=2, choices=[0, 1, 2], help="평가 환경 단계")
    parser.add_argument("--seed", type=int, default=100, help="평가 난수 시드")
    parser.add_argument("--max-steps", type=int, default=5_000, help="에피소드 최대 스텝 수")
    parser.add_argument("--render", action="store_true", help="PPO 평가 중 터미널 보드 렌더링 사용")
    parser.add_argument("--skip-heuristic", action="store_true", help="휴리스틱 기준선 평가를 건너뜁니다")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "tetris_rl" / "logs",
        help="그래프와 비교 리포트 저장 경로",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "tetris_rl" / "models",
        help="기본 모델 조회 경로",
    )
    return parser.parse_args()


def find_default_model(model_dir: Path) -> Path:
    preferred_names = [
        "tetris_ppo_final.zip",
        "tetris_ppo_stage2.zip",
        "stage2.zip",
        "tetris_ppo_latest.zip",
        "tetris_ppo_stage1.zip",
        "stage1.zip",
        "tetris_ppo_stage0.zip",
        "stage0.zip",
    ]
    for name in preferred_names:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    for stage in (2, 1, 0):
        candidate = model_dir / f"stage{stage}.zip"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("평가할 모델을 찾을 수 없습니다. --model 경로를 지정하거나 먼저 학습을 실행하세요.")


def configure_korean_font() -> None:
    """가능한 경우 윈도우 기본 한글 글꼴을 그래프에 적용합니다."""
    try:
        from matplotlib import font_manager

        font_names = {font.name for font in font_manager.fontManager.ttflist}
        if "Malgun Gothic" in font_names:
            plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def load_training_curve(log_dir: Path) -> tuple[list[int], list[float]]:
    """TensorBoard 로그에서 학습 보상 곡선을 읽습니다."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return [], []

    steps: list[int] = []
    rewards: list[float] = []
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", [])
            tag = "rollout/ep_rew_mean" if "rollout/ep_rew_mean" in scalar_tags else None
            if tag is None:
                continue
            for scalar in accumulator.Scalars(tag):
                steps.append(int(scalar.step))
                rewards.append(float(scalar.value))
        except Exception:
            continue

    if not steps:
        return [], []

    ordered = sorted(zip(steps, rewards), key=lambda item: item[0])
    sorted_steps, sorted_rewards = zip(*ordered)
    return list(sorted_steps), list(sorted_rewards)


def run_policy(
    policy_name: str,
    env: TetrisEnv,
    episodes: int,
    seed: int,
    action_policy: ActionPolicy,
    render: bool = False,
) -> EvaluationResult:
    survival_steps: list[int] = []
    line_counts: list[int] = []
    episode_rewards: list[float] = []

    print(f"{policy_name} 평가를 시작합니다. 에피소드 수: {episodes}")

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        last_info: dict[str, Any] = {"steps": 0, "lines_cleared": 0}

        while not done:
            action = action_policy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            last_info = info
            if render:
                env.render()

        survival_steps.append(int(last_info["steps"]))
        line_counts.append(int(last_info["lines_cleared"]))
        episode_rewards.append(total_reward)

    return EvaluationResult(
        policy_name=policy_name,
        survival_steps=survival_steps,
        line_counts=line_counts,
        rewards=episode_rewards,
    )


def calculate_improvement(ppo_summary: dict[str, float], baseline_summary: dict[str, float]) -> dict[str, dict[str, float | None]]:
    """PPO가 휴리스틱 대비 얼마나 개선됐는지 계산합니다."""
    improvement: dict[str, dict[str, float | None]] = {}
    for metric, ppo_value in ppo_summary.items():
        baseline_value = baseline_summary[metric]
        difference = ppo_value - baseline_value
        if abs(baseline_value) < 1e-9:
            improvement_percent = None
        else:
            improvement_percent = difference / abs(baseline_value) * 100.0

        improvement[metric] = {
            "ppo": ppo_value,
            "heuristic": baseline_value,
            "difference": difference,
            "improvement_percent": improvement_percent,
        }
    return improvement


def save_episode_metrics(log_dir: Path, results: list[EvaluationResult]) -> Path:
    """에피소드별 평가 결과를 CSV로 저장합니다."""
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / "evaluation_episodes.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["policy", "episode", "survival_steps", "lines_cleared", "reward"])
        for result in results:
            for index, (steps, lines, reward) in enumerate(
                zip(result.survival_steps, result.line_counts, result.rewards),
                start=1,
            ):
                writer.writerow([result.policy_name, index, steps, lines, f"{reward:.6f}"])
    return output_path


def save_comparison_report(
    log_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
    ppo_result: EvaluationResult,
    heuristic_result: EvaluationResult | None,
) -> tuple[Path, Path | None]:
    """평균 성능과 개선율을 JSON/CSV 리포트로 저장합니다."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ppo_summary = ppo_result.summary()

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "episodes": args.episodes,
        "stage": args.stage,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "ppo": ppo_summary,
        "heuristic_policy": "한 스텝 배치 후보를 라인 클리어, 구멍 수, 총 높이, 굴곡도, 게임오버 패널티로 평가",
    }

    csv_path: Path | None = None
    if heuristic_result is not None:
        heuristic_summary = heuristic_result.summary()
        improvement = calculate_improvement(ppo_summary, heuristic_summary)
        report["heuristic"] = heuristic_summary
        report["improvement_vs_heuristic"] = improvement

        csv_path = log_dir / "performance_comparison.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["metric", "ppo", "heuristic", "difference", "improvement_percent"])
            for metric, values in improvement.items():
                percent = values["improvement_percent"]
                writer.writerow(
                    [
                        metric,
                        f"{values['ppo']:.6f}",
                        f"{values['heuristic']:.6f}",
                        f"{values['difference']:.6f}",
                        "" if percent is None else f"{percent:.6f}",
                    ]
                )

    json_path = log_dir / "performance_comparison.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return json_path, csv_path


def save_reward_curve(
    log_dir: Path,
    ppo_result: EvaluationResult,
    heuristic_result: EvaluationResult | None,
) -> Path:
    configure_korean_font()
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / "reward_curve.png"

    train_steps, train_rewards = load_training_curve(log_dir)

    plt.figure(figsize=(10, 5))
    if train_steps and train_rewards:
        plt.plot(train_steps, train_rewards, label="학습 평균 보상")
        plt.axhline(np.mean(ppo_result.rewards), color="tab:green", linestyle="--", label="PPO 평가 평균 보상")
        if heuristic_result is not None:
            plt.axhline(
                np.mean(heuristic_result.rewards),
                color="tab:red",
                linestyle=":",
                label="휴리스틱 평가 평균 보상",
            )
        plt.xlabel("학습 스텝")
        plt.ylabel("평균 보상")
        plt.title("테트리스 PPO 학습 및 휴리스틱 비교")
    else:
        episode_axis = np.arange(1, len(ppo_result.rewards) + 1)
        plt.plot(episode_axis, ppo_result.rewards, label="PPO 평가 보상")
        if heuristic_result is not None:
            plt.plot(episode_axis, heuristic_result.rewards, label="휴리스틱 평가 보상")
        plt.xlabel("에피소드")
        plt.ylabel("보상")
        plt.title("테트리스 PPO 평가 보상 비교")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def print_summary(result: EvaluationResult) -> None:
    summary = result.summary()
    print(f"{result.policy_name} 평균 생존 스텝: {summary['average_survival_steps']:.2f}")
    print(f"{result.policy_name} 평균 라인 클리어 수: {summary['average_lines_cleared']:.2f}")
    print(f"{result.policy_name} 평균 보상: {summary['average_reward']:.2f}")


def print_improvement(improvement: dict[str, dict[str, float | None]]) -> None:
    labels = {
        "average_survival_steps": "생존 스텝",
        "average_lines_cleared": "라인 클리어 수",
        "average_reward": "보상",
    }
    for metric, values in improvement.items():
        percent = values["improvement_percent"]
        percent_text = "계산 불가" if percent is None else f"{percent:.2f}%"
        print(f"PPO의 휴리스틱 대비 평균 {labels[metric]} 개선율: {percent_text}")


def evaluate_model(args: argparse.Namespace) -> None:
    model_path = args.model if args.model else find_default_model(args.model_dir)
    model = PPO.load(str(model_path))

    def ppo_policy(obs: np.ndarray, _env: TetrisEnv) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(np.asarray(action).item())

    print(f"모델을 불러왔습니다: {model_path}")

    ppo_env = TetrisEnv(stage=args.stage, max_steps=args.max_steps, render_mode="human" if args.render else None)
    ppo_result = run_policy("PPO", ppo_env, args.episodes, args.seed, ppo_policy, render=args.render)
    ppo_env.close()

    heuristic_result: EvaluationResult | None = None
    if not args.skip_heuristic:
        heuristic_env = TetrisEnv(stage=args.stage, max_steps=args.max_steps)
        heuristic_result = run_policy("휴리스틱", heuristic_env, args.episodes, args.seed, choose_heuristic_action)
        heuristic_env.close()

    curve_path = save_reward_curve(args.log_dir, ppo_result, heuristic_result)
    episode_path = save_episode_metrics(args.log_dir, [result for result in [ppo_result, heuristic_result] if result is not None])
    json_path, csv_path = save_comparison_report(args.log_dir, model_path, args, ppo_result, heuristic_result)

    print_summary(ppo_result)
    if heuristic_result is not None:
        print_summary(heuristic_result)
        print_improvement(calculate_improvement(ppo_result.summary(), heuristic_result.summary()))

    print(f"에피소드별 평가 기록 저장 완료: {episode_path}")
    print(f"성능 비교 JSON 저장 완료: {json_path}")
    if csv_path is not None:
        print(f"성능 비교 CSV 저장 완료: {csv_path}")
    print(f"보상 곡선 저장 완료: {curve_path}")


def main() -> None:
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
