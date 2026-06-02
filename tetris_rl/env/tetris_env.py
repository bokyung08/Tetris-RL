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


STAGE_CONFIGS = {
    0: StageConfig(blocks=("I", "O"), fall_speed=1.0, expose_next_block=False),
    1: StageConfig(blocks=BLOCK_NAMES, fall_speed=0.5, expose_next_block=False),
    2: StageConfig(blocks=BLOCK_NAMES, fall_speed=0.25, expose_next_block=True),
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
    ) -> None:
        super().__init__()
        self.width = BOARD_WIDTH
        self.height = BOARD_HEIGHT
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.width * 4)

        # 명세에는 32차원이라고 되어 있지만, 지정된 항목 합계는 35차원입니다.
        # 항목을 누락하지 않기 위해 10+10+7+7+1 구조를 그대로 사용합니다.
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
        """커리큘럼 학습 단계에 맞춰 블록 풀과 next block 노출 여부를 바꿉니다."""
        if stage not in STAGE_CONFIGS:
            raise ValueError("지원하지 않는 커리큘럼 단계입니다. stage는 0, 1, 2 중 하나여야 합니다.")
        self.stage = stage
        self.stage_config = STAGE_CONFIGS[stage]

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

        column, rotation = self._decode_action(action)
        matrix = BLOCK_ROTATIONS[self.current_block][rotation]
        placed_column = self._fit_column(column, matrix)

        holes_before = int(np.sum(self._get_holes()))
        placement_failed, spawn_collision = self._drop_and_lock(matrix, placed_column)

        cleared_lines = 0
        if not placement_failed:
            cleared_lines = self._clear_lines()

        heights = self._get_column_heights()
        holes_after = int(np.sum(self._get_holes()))
        new_holes = max(0, holes_after - holes_before)
        aggregate_height = int(np.sum(heights))
        bumpiness = int(np.sum(np.abs(np.diff(heights))))

        reward = self._calculate_reward(
            cleared_lines=cleared_lines,
            new_holes=new_holes,
            aggregate_height=aggregate_height,
            bumpiness=bumpiness,
            game_over=placement_failed or spawn_collision,
        )

        self.steps += 1
        self.total_lines += cleared_lines
        self.total_reward += reward
        self.game_over = placement_failed or spawn_collision

        self.current_block = self.next_block
        self.next_block = self._sample_block()

        terminated = self.game_over
        truncated = self.steps >= self.max_steps and not terminated

        info = self._get_info(
            cleared_lines=cleared_lines,
            requested_column=column,
            placed_column=placed_column,
            rotation=rotation,
            new_holes=new_holes,
            aggregate_height=aggregate_height,
            bumpiness=bumpiness,
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def preview_action(self, action: int) -> dict[str, Any]:
        """환경 상태를 바꾸지 않고 특정 행동의 즉시 배치 결과를 계산합니다."""
        original_board = self.board.copy()

        try:
            column, rotation = self._decode_action(action)
            matrix = BLOCK_ROTATIONS[self.current_block][rotation]
            placed_column = self._fit_column(column, matrix)

            holes_before = int(np.sum(self._get_holes()))
            placement_failed, spawn_collision = self._drop_and_lock(matrix, placed_column)

            cleared_lines = 0
            if not placement_failed:
                cleared_lines = self._clear_lines()

            heights = self._get_column_heights()
            holes_after = int(np.sum(self._get_holes()))
            new_holes = max(0, holes_after - holes_before)
            aggregate_height = int(np.sum(heights))
            bumpiness = int(np.sum(np.abs(np.diff(heights))))
            game_over = placement_failed or spawn_collision
            reward = self._calculate_reward(
                cleared_lines=cleared_lines,
                new_holes=new_holes,
                aggregate_height=aggregate_height,
                bumpiness=bumpiness,
                game_over=game_over,
            )

            return {
                "action": int(action),
                "requested_column": column,
                "placed_column": placed_column,
                "rotation": rotation,
                "cleared_lines": cleared_lines,
                "new_holes": new_holes,
                "total_holes": holes_after,
                "aggregate_height": aggregate_height,
                "bumpiness": bumpiness,
                "game_over": game_over,
                "reward": float(reward),
            }
        finally:
            self.board = original_board

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
        heights = self._get_column_heights()
        holes = self._get_holes()
        current_block = self._one_hot_block(self.current_block)
        if self.stage_config.expose_next_block:
            next_block = self._one_hot_block(self.next_block)
        else:
            next_block = np.zeros(len(BLOCK_NAMES), dtype=np.float32)
        stage = np.array([float(self.stage)], dtype=np.float32)
        return np.concatenate([heights, holes, current_block, next_block, stage]).astype(np.float32)

    def _calculate_reward(
        self,
        *,
        cleared_lines: int,
        new_holes: int,
        aggregate_height: int,
        bumpiness: int,
        game_over: bool,
    ) -> float:
        reward = 100.0 * (cleared_lines**2)
        reward -= 5.0 * new_holes
        reward -= 0.5 * aggregate_height
        reward -= 0.3 * bumpiness
        reward += 0.1
        if game_over:
            reward -= 50.0
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
