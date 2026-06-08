from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch.utils.data import DataLoader, TensorDataset

from tetris_hu.heuristic_policy import score_preview
from tetris_rl.env.tetris_env import TetrisEnv


@dataclass
class ImitationDataset:
    observations: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    stages: np.ndarray

    def stage_counts(self) -> dict[int, int]:
        """stage별 샘플 수를 반환합니다."""
        unique_stages, counts = np.unique(self.stages, return_counts=True)
        return {int(stage): int(count) for stage, count in zip(unique_stages, counts)}


def parse_stages(raw_stages: str) -> list[int]:
    stages = [int(stage.strip()) for stage in raw_stages.split(",") if stage.strip()]
    invalid_stages = [stage for stage in stages if stage not in (0, 1, 2)]
    if invalid_stages:
        raise ValueError("stage는 0, 1, 2만 사용할 수 있습니다.")
    return stages


def parse_stage_sample_counts(raw_counts: str | None, stages: list[int], default_samples: int) -> dict[int, int]:
    """`0:5000,1:20000` 형식의 stage별 샘플 수 설정을 파싱합니다."""
    counts = {stage: default_samples for stage in stages}
    if raw_counts is None or not raw_counts.strip():
        return counts

    for item in raw_counts.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("stage별 샘플 수는 '0:5000,1:20000,2:20000' 형식이어야 합니다.")
        raw_stage, raw_count = item.split(":", maxsplit=1)
        stage = int(raw_stage.strip())
        count = int(raw_count.strip())
        if stage not in (0, 1, 2):
            raise ValueError("stage는 0, 1, 2만 사용할 수 있습니다.")
        if count < 0:
            raise ValueError("stage별 샘플 수는 0 이상이어야 합니다.")
        counts[stage] = count

    return {stage: counts[stage] for stage in stages}


def choose_masked_heuristic_action(env: TetrisEnv, action_mask: np.ndarray) -> int:
    """합법 행동만 대상으로 휴리스틱 최고 점수 행동을 고릅니다."""
    best_action = 0
    best_score = -float("inf")
    for action in np.flatnonzero(action_mask):
        score = score_preview(env.preview_action(int(action)))
        if score > best_score:
            best_score = score
            best_action = int(action)
    return best_action


def collect_imitation_dataset(
    *,
    stages: list[int],
    samples_per_stage: int,
    max_steps: int,
    seed: int,
    samples_by_stage: dict[int, int] | None = None,
) -> ImitationDataset:
    """휴리스틱 정책으로 관측, 행동, action mask 데이터셋을 수집합니다."""
    observations: list[np.ndarray] = []
    actions: list[int] = []
    action_masks: list[np.ndarray] = []
    sample_stages: list[int] = []

    for stage in stages:
        target_samples = samples_by_stage.get(stage, samples_per_stage) if samples_by_stage else samples_per_stage
        if target_samples <= 0:
            print(f"Stage {stage} 데이터 수집을 건너뜁니다.")
            continue

        env = TetrisEnv(stage=stage, max_steps=max_steps)
        stage_samples = 0
        episode = 0

        while stage_samples < target_samples:
            obs, _ = env.reset(seed=seed + stage * 100_000 + episode)
            done = False

            while not done and stage_samples < target_samples:
                mask = env.action_masks()
                action = choose_masked_heuristic_action(env, mask)
                observations.append(obs.copy())
                actions.append(action)
                action_masks.append(mask.copy())
                sample_stages.append(stage)

                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                stage_samples += 1

                if stage_samples % 1_000 == 0:
                    print(f"Stage {stage} 데이터 수집: {stage_samples}/{target_samples}")

            episode += 1

        env.close()
        print(f"Stage {stage} 데이터 수집 완료: {stage_samples}개")

    return ImitationDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        action_masks=np.asarray(action_masks, dtype=bool),
        stages=np.asarray(sample_stages, dtype=np.int64),
    )


def evaluate_behavior_cloning_accuracy(
    *,
    model: MaskablePPO,
    dataset: ImitationDataset,
    batch_size: int,
) -> tuple[float, dict[int, float]]:
    """모델이 휴리스틱 행동을 얼마나 맞히는지 전체/stage별로 계산합니다."""
    device = model.device
    total_correct = 0
    total_samples = 0
    stage_correct = {stage: 0 for stage in (0, 1, 2)}
    stage_total = {stage: 0 for stage in (0, 1, 2)}

    was_training = model.policy.training
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(dataset.actions), batch_size):
            end = min(start + batch_size, len(dataset.actions))
            obs_batch = torch.as_tensor(dataset.observations[start:end], dtype=torch.float32).to(device)
            action_batch = torch.as_tensor(dataset.actions[start:end], dtype=torch.long).to(device)
            mask_batch = dataset.action_masks[start:end]
            stage_batch = dataset.stages[start:end]

            distribution = model.policy.get_distribution(obs_batch, action_masks=mask_batch)
            predicted_actions = distribution.mode()
            correct = (predicted_actions == action_batch).detach().cpu().numpy()

            total_correct += int(correct.sum())
            total_samples += int(correct.shape[0])
            for stage in (0, 1, 2):
                stage_mask = stage_batch == stage
                if not np.any(stage_mask):
                    continue
                stage_total[stage] += int(stage_mask.sum())
                stage_correct[stage] += int(correct[stage_mask].sum())

    overall_accuracy = total_correct / max(1, total_samples) * 100.0
    stage_accuracy = {
        stage: stage_correct[stage] / stage_total[stage] * 100.0
        for stage in (0, 1, 2)
        if stage_total[stage] > 0
    }
    model.policy.set_training_mode(was_training)
    return overall_accuracy, stage_accuracy


def train_behavior_cloning(
    *,
    model: MaskablePPO,
    dataset: ImitationDataset,
    epochs: int,
    batch_size: int,
    entropy_coef: float,
) -> tuple[float, float]:
    """휴리스틱 행동을 맞히도록 policy actor를 지도학습합니다."""
    device = model.device
    tensor_dataset = TensorDataset(
        torch.as_tensor(dataset.observations, dtype=torch.float32),
        torch.as_tensor(dataset.actions, dtype=torch.long),
        torch.as_tensor(dataset.action_masks, dtype=torch.bool),
    )
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.policy.set_training_mode(True)
    last_average_loss = 0.0
    last_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for obs_batch, action_batch, mask_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            mask_batch_np = mask_batch.cpu().numpy()

            distribution = model.policy.get_distribution(obs_batch, action_masks=mask_batch_np)
            log_prob = distribution.log_prob(action_batch)
            entropy = distribution.entropy().mean()
            loss = -log_prob.mean() - entropy_coef * entropy

            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            model.policy.optimizer.step()

            with torch.no_grad():
                predicted_actions = distribution.mode()
                total_correct += int((predicted_actions == action_batch).sum().item())
                total_samples += int(action_batch.shape[0])
                total_loss += float(loss.item()) * int(action_batch.shape[0])

        last_average_loss = total_loss / max(1, total_samples)
        last_accuracy = total_correct / max(1, total_samples) * 100.0
        print(f"BC 정규화 epoch {epoch}/{epochs}: loss={last_average_loss:.4f}, 행동 일치율={last_accuracy:.2f}%")

    return last_average_loss, last_accuracy
