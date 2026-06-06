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


def parse_stages(raw_stages: str) -> list[int]:
    stages = [int(stage.strip()) for stage in raw_stages.split(",") if stage.strip()]
    invalid_stages = [stage for stage in stages if stage not in (0, 1, 2)]
    if invalid_stages:
        raise ValueError("stage는 0, 1, 2만 사용할 수 있습니다.")
    return stages


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
) -> ImitationDataset:
    """휴리스틱 정책으로 관측, 행동, action mask 데이터셋을 수집합니다."""
    observations: list[np.ndarray] = []
    actions: list[int] = []
    action_masks: list[np.ndarray] = []

    for stage in stages:
        env = TetrisEnv(stage=stage, max_steps=max_steps)
        stage_samples = 0
        episode = 0

        while stage_samples < samples_per_stage:
            obs, _ = env.reset(seed=seed + stage * 100_000 + episode)
            done = False

            while not done and stage_samples < samples_per_stage:
                mask = env.action_masks()
                action = choose_masked_heuristic_action(env, mask)
                observations.append(obs.copy())
                actions.append(action)
                action_masks.append(mask.copy())

                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                stage_samples += 1

                if stage_samples % 1_000 == 0:
                    print(f"Stage {stage} 데이터 수집: {stage_samples}/{samples_per_stage}")

            episode += 1

        env.close()
        print(f"Stage {stage} 데이터 수집 완료: {stage_samples}개")

    return ImitationDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        action_masks=np.asarray(action_masks, dtype=bool),
    )


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
