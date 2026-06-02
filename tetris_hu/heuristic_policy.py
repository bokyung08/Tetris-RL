from __future__ import annotations

from typing import Any

import numpy as np


def score_preview(preview: dict[str, Any]) -> float:
    """한 스텝 배치 미리보기 결과를 휴리스틱 점수로 변환합니다."""
    score = 120.0 * (preview["cleared_lines"] ** 2)
    score -= 12.0 * preview["total_holes"]
    score -= 0.4 * preview["aggregate_height"]
    score -= 0.5 * preview["bumpiness"]
    score -= 1.0 * max(0, preview["max_height"] - 14) ** 2
    if preview["invalid_action"]:
        score -= 10.0
    if preview["game_over"]:
        score -= 1_000.0
    return score


def choose_heuristic_action(_obs: np.ndarray, env: Any) -> int:
    """학습 없이 40개 즉시 배치 후보를 비교하는 단순 휴리스틱 정책입니다."""
    best_action = 0
    best_score = -float("inf")

    for action in range(env.action_space.n):
        score = score_preview(env.preview_action(action))
        if score > best_score:
            best_score = score
            best_action = action

    return best_action
