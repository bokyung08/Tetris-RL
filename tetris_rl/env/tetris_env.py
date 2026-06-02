from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
BLOCK_TO_INDEX = {name: index for index, name in enumerate(BLOCK_NAMES)}


@dataclass(frozen=True)
class StageConfig:
    blocks: tuple[str, ...]
    fall_speed: float
    expose_next_block: bool


@dataclass(frozen=True)
class BoardMetrics:
    heights: np.ndarray
    holes: np.ndarray
    aggregate_height: int
    total_holes: int
    bumpiness: int
    max_height: int


@dataclass(frozen=True)
class RewardConfig:
    survival_bonus: float = 2.0
    safe_placement_bonus: float = 1.0
    line_rewards: tuple[float, float, float, float, float] = (0.0, 40.0, 120.0, 300.0, 800.0)
    new_hole_penalty: float = 10.0
    hole_delta_penalty: float = 6.0
    height_increase_penalty: float = 0.3
    height_decrease_bonus: float = 0.2
    bumpiness_increase_penalty: float = 0.4
    bumpiness_decrease_bonus: float = 0.2
    danger_height: int = 16
    danger_height_penalty: float = 0.4
    invalid_action_penalty: float = 2.0
    game_over_penalty: float = 80.0


STAGE_CONFIGS = {
    0: StageConfig(blocks=("I", "O"), fall_speed=1.0, expose_next_block=False),
    1: StageConfig(blocks=BLOCK_NAMES, fall_speed=0.5, expose_next_block=False),
    2: StageConfig(blocks=BLOCK_NAMES, fall_speed=0.25, expose_next_block=True),
}


STAGE_REWARD_CONFIGS = {
    0: RewardConfig(
        survival_bonus=2.0,
        safe_placement_bonus=1.2,
        line_rewards=(0.0, 35.0, 100.0, 240.0, 650.0),
        new_hole_penalty=4.0,
        hole_delta_penalty=2.0,
        height_increase_penalty=0.15,
        bumpiness_increase_penalty=0.2,
        danger_height=17,
        danger_height_penalty=0.15,
        game_over_penalty=80.0,
    ),
    1: RewardConfig(
        survival_bonus=2.0,
        safe_placement_bonus=1.0,
        line_rewards=(0.0, 40.0, 120.0, 300.0, 800.0),
        new_hole_penalty=8.0,
        hole_delta_penalty=4.0,
        height_increase_penalty=0.25,
        bumpiness_increase_penalty=0.35,
        danger_height=16,
        danger_height_penalty=0.3,
        game_over_penalty=80.0,
    ),
    2: RewardConfig(
        survival_bonus=1.5,
        safe_placement_bonus=0.8,
        line_rewards=(0.0, 45.0, 140.0, 340.0, 900.0),
        new_hole_penalty=12.0,
        hole_delta_penalty=8.0,
        height_increase_penalty=0.35,
        bumpiness_increase_penalty=0.5,
        danger_height=15,
        danger_height_penalty=0.5,
        game_over_penalty=100.0,
    ),
}


BASE_BLOCKS = {
    "I": np.array([[1, 1, 1, 1]], dtype=np.int8),
    "O": np.array([[1, 1], [1, 1]], dtype=np.int8),
    "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8),
    "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),
    "Z": np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
    "J": np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int8),
    "L": np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int8),
}


def _trim_matrix(matrix: np.ndarray) -> np.ndarray:
    """회전 후 생긴 빈 행과 빈 열을 제거합니다."""
    rows = np.where(matrix.any(axis=1))[0]
    cols = np.where(matrix.any(axis=0))[0]
    return matrix[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _build_rotations(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """블록별 4개 회전 상태를 생성합니다."""
    return tuple(_trim_matrix(np.rot90(matrix, -rotation)).astype(np.int8) for rotation in range(4))


BLOCK_ROTATIONS = {name: _build_rotations(matrix) for name, matrix in BASE_BLOCKS.items()}


class TetrisEnv(gym.Env):
    """즉시 배치 방식의 테트리스 강화학습 환경입니다."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        stage: int = 0,
        max_steps: int = 5_000,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.width = BOARD_WIDTH
        self.height = BOARD_HEIGHT
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.custom_reward_config = reward_config
        self.reward_config = reward_config or STAGE_REWARD_CONFIGS[stage]

        self.action_space = spaces.Discrete(self.width * 4)

        # 지정된 상태 항목 합계는 10+10+7+7+1=35차원입니다.
        self.observation_size = self.width + self.width + len(BLOCK_NAMES) + len(BLOCK_NAMES) + 1
        low = np.zeros(self.observation_size, dtype=np.float32)
        high = np.concatenate(
            [
                np.full(self.width, self.height, dtype=np.float32),
                np.full(self.width, self.height, dtype=np.float32),
                np.ones(len(BLOCK_NAMES), dtype=np.float32),
                np.ones(len(BLOCK_NAMES), dtype=np.float32),
                np.array([2.0], dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.current_block = "I"
        self.next_block = "O"
        self.steps = 0
        self.total_lines = 0
        self.total_reward = 0.0
        self.game_over = False
        self.set_stage(stage)

    def set_stage(self, stage: int) -> None:
        """커리큘럼 단계에 맞춰 블록 풀과 next block 노출 여부를 바꿉니다."""
        if stage not in STAGE_CONFIGS:
            raise ValueError("지원하지 않는 커리큘럼 단계입니다. stage는 0, 1, 2 중 하나여야 합니다.")
        self.stage = stage
        self.stage_config = STAGE_CONFIGS[stage]
        if self.custom_reward_config is None:
            self.reward_config = STAGE_REWARD_CONFIGS[stage]

    def get_stage(self) -> int:
        """현재 커리큘럼 단계를 반환합니다."""
        return self.stage

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if options and "stage" in options:
            self.set_stage(int(options["stage"]))

        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.current_block = self._sample_block()
        self.next_block = self._sample_block()
        self.steps = 0
        self.total_lines = 0
        self.total_reward = 0.0
        self.game_over = False
        return self._get_observation(), self._get_info(cleared_lines=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info(cleared_lines=0)

        result = self._place_action(action)
        reward = result["reward"]

        self.steps += 1
        self.total_lines += int(result["cleared_lines"])
        self.total_reward += reward
        self.game_over = bool(result["game_over"])

        self.current_block = self.next_block
        self.next_block = self._sample_block()

        terminated = self.game_over
        truncated = self.steps >= self.max_steps and not terminated

        info = self._get_info(
            cleared_lines=int(result["cleared_lines"]),
            requested_column=int(result["requested_column"]),
            placed_column=int(result["placed_column"]),
            rotation=int(result["rotation"]),
            invalid_action=bool(result["invalid_action"]),
            new_holes=int(result["new_holes"]),
            hole_delta=int(result["hole_delta"]),
            aggregate_height=int(result["aggregate_height"]),
            height_delta=int(result["height_delta"]),
            bumpiness=int(result["bumpiness"]),
            bumpiness_delta=int(result["bumpiness_delta"]),
            max_height=int(result["max_height"]),
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def preview_action(self, action: int) -> dict[str, Any]:
        """환경 상태를 바꾸지 않고 특정 행동의 즉시 배치 결과를 계산합니다."""
        original_board = self.board.copy()
        try:
            return self._place_action(action)
        finally:
            self.board = original_board

    def get_action_mask(self) -> np.ndarray:
        """현재 블록 기준으로 보드 폭을 넘는 중복 행동을 표시합니다."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action in range(self.action_space.n):
            column, rotation = self._decode_action(action)
            matrix = BLOCK_ROTATIONS[self.current_block][rotation]
            mask[action] = column <= self.width - matrix.shape[1]
        return mask

    def render(self) -> str | None:
        lines = ["+" + "-" * self.width + "+"]
        for row in self.board:
            lines.append("|" + "".join("#" if cell else "." for cell in row) + "|")
        lines.append("+" + "-" * self.width + "+")
        lines.append(f"현재 블록: {self.current_block} / 다음 블록: {self.next_block} / 단계: {self.stage}")
        rendered = "\n".join(lines)

        if self.render_mode == "ansi":
            return rendered

        print("현재 보드:")
        print(rendered)
        return None

    def close(self) -> None:
        """Gymnasium 인터페이스 호환을 위한 종료 메서드입니다."""

    def _place_action(self, action: int) -> dict[str, Any]:
        metrics_before = self._get_board_metrics()
        column, rotation = self._decode_action(action)
        matrix = BLOCK_ROTATIONS[self.current_block][rotation]
        placed_column = self._fit_column(column, matrix)
        invalid_action = column != placed_column

        placement_failed, spawn_collision = self._drop_and_lock(matrix, placed_column)

        cleared_lines = 0
        if not placement_failed:
            cleared_lines = self._clear_lines()

        metrics_after = self._get_board_metrics()
        game_over = placement_failed or spawn_collision
        reward = self._calculate_reward(
            cleared_lines=cleared_lines,
            before=metrics_before,
            after=metrics_after,
            invalid_action=invalid_action,
            game_over=game_over,
        )

        return {
            "action": int(action),
            "requested_column": column,
            "placed_column": placed_column,
            "rotation": rotation,
            "invalid_action": invalid_action,
            "cleared_lines": cleared_lines,
            "new_holes": max(0, metrics_after.total_holes - metrics_before.total_holes),
            "hole_delta": metrics_after.total_holes - metrics_before.total_holes,
            "total_holes": metrics_after.total_holes,
            "aggregate_height": metrics_after.aggregate_height,
            "height_delta": metrics_after.aggregate_height - metrics_before.aggregate_height,
            "bumpiness": metrics_after.bumpiness,
            "bumpiness_delta": metrics_after.bumpiness - metrics_before.bumpiness,
            "max_height": metrics_after.max_height,
            "game_over": game_over,
            "reward": float(reward),
        }

    def _decode_action(self, action: int) -> tuple[int, int]:
        action = int(action)
        column = action // 4
        rotation = action % 4
        return column, rotation

    def _sample_block(self) -> str:
        return str(self.np_random.choice(self.stage_config.blocks))

    def _fit_column(self, column: int, matrix: np.ndarray) -> int:
        """블록 폭이 보드를 넘지 않도록 선택 열을 합법 범위로 보정합니다."""
        max_column = self.width - matrix.shape[1]
        return int(np.clip(column, 0, max_column))

    def _can_place(self, matrix: np.ndarray, top_row: int, left_col: int) -> bool:
        for row_offset, col_offset in np.argwhere(matrix == 1):
            row = top_row + int(row_offset)
            col = left_col + int(col_offset)
            if col < 0 or col >= self.width or row >= self.height:
                return False
            if row >= 0 and self.board[row, col] == 1:
                return False
        return True

    def _drop_and_lock(self, matrix: np.ndarray, left_col: int) -> tuple[bool, bool]:
        top_row = -matrix.shape[0]
        while self._can_place(matrix, top_row + 1, left_col):
            top_row += 1

        if not self._can_place(matrix, top_row, left_col):
            return True, True

        spawn_collision = False
        for row_offset, col_offset in np.argwhere(matrix == 1):
            row = top_row + int(row_offset)
            col = left_col + int(col_offset)
            if row < 0:
                spawn_collision = True
            else:
                self.board[row, col] = 1
        return False, spawn_collision

    def _clear_lines(self) -> int:
        full_rows = np.all(self.board == 1, axis=1)
        cleared_lines = int(np.sum(full_rows))
        if cleared_lines == 0:
            return 0

        remaining_rows = self.board[~full_rows]
        empty_rows = np.zeros((cleared_lines, self.width), dtype=np.int8)
        self.board = np.vstack([empty_rows, remaining_rows])
        return cleared_lines

    def _get_board_metrics(self) -> BoardMetrics:
        heights = self._get_column_heights()
        holes = self._get_holes()
        aggregate_height = int(np.sum(heights))
        total_holes = int(np.sum(holes))
        bumpiness = int(np.sum(np.abs(np.diff(heights))))
        max_height = int(np.max(heights))
        return BoardMetrics(
            heights=heights,
            holes=holes,
            aggregate_height=aggregate_height,
            total_holes=total_holes,
            bumpiness=bumpiness,
            max_height=max_height,
        )

    def _get_column_heights(self) -> np.ndarray:
        heights = np.zeros(self.width, dtype=np.float32)
        for column in range(self.width):
            filled_rows = np.where(self.board[:, column] == 1)[0]
            if filled_rows.size > 0:
                heights[column] = self.height - filled_rows[0]
        return heights

    def _get_holes(self) -> np.ndarray:
        holes = np.zeros(self.width, dtype=np.float32)
        for column in range(self.width):
            filled_rows = np.where(self.board[:, column] == 1)[0]
            if filled_rows.size == 0:
                continue
            first_filled = filled_rows[0]
            holes[column] = np.sum(self.board[first_filled:, column] == 0)
        return holes

    def _one_hot_block(self, block_name: str) -> np.ndarray:
        vector = np.zeros(len(BLOCK_NAMES), dtype=np.float32)
        vector[BLOCK_TO_INDEX[block_name]] = 1.0
        return vector

    def _get_observation(self) -> np.ndarray:
        metrics = self._get_board_metrics()
        current_block = self._one_hot_block(self.current_block)
        if self.stage_config.expose_next_block:
            next_block = self._one_hot_block(self.next_block)
        else:
            next_block = np.zeros(len(BLOCK_NAMES), dtype=np.float32)
        stage = np.array([float(self.stage)], dtype=np.float32)
        return np.concatenate([metrics.heights, metrics.holes, current_block, next_block, stage]).astype(np.float32)

    def _calculate_reward(
        self,
        *,
        cleared_lines: int,
        before: BoardMetrics,
        after: BoardMetrics,
        invalid_action: bool,
        game_over: bool,
    ) -> float:
        config = self.reward_config
        hole_delta = after.total_holes - before.total_holes
        height_delta = after.aggregate_height - before.aggregate_height
        bumpiness_delta = after.bumpiness - before.bumpiness
        danger_height = max(0, after.max_height - config.danger_height)

        reward = config.survival_bonus
        reward += config.line_rewards[min(cleared_lines, 4)]
        reward += config.safe_placement_bonus if hole_delta <= 0 and not game_over else 0.0
        reward -= config.new_hole_penalty * max(0, hole_delta)
        reward -= config.hole_delta_penalty * max(0, hole_delta)
        reward -= config.height_increase_penalty * max(0, height_delta)
        reward += config.height_decrease_bonus * max(0, -height_delta)
        reward -= config.bumpiness_increase_penalty * max(0, bumpiness_delta)
        reward += config.bumpiness_decrease_bonus * max(0, -bumpiness_delta)
        reward -= config.danger_height_penalty * (danger_height**2)

        if invalid_action:
            reward -= config.invalid_action_penalty
        if game_over:
            reward -= config.game_over_penalty
        return reward

    def _get_info(self, cleared_lines: int, **kwargs: Any) -> dict[str, Any]:
        info = {
            "stage": self.stage,
            "fall_speed": self.stage_config.fall_speed,
            "block": self.current_block,
            "next_block": self.next_block,
            "steps": self.steps,
            "lines_cleared": self.total_lines,
            "cleared_lines": cleared_lines,
            "total_reward": self.total_reward,
            "game_over": self.game_over,
        }
        info.update(kwargs)
        return info
