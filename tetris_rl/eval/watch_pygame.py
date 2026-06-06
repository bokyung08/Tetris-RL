from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pygame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_hu import choose_heuristic_action
from tetris_rl.env.tetris_env import BLOCK_ROTATIONS, TetrisEnv
from tetris_rl.eval.evaluate import find_default_model, load_policy_model


BOARD_COLORS = {
    0: (24, 28, 37),
    1: (78, 181, 183),
}
BLOCK_COLORS = {
    "I": (74, 222, 255),
    "O": (255, 214, 102),
    "T": (180, 126, 255),
    "S": (91, 214, 128),
    "Z": (255, 110, 110),
    "J": (93, 150, 255),
    "L": (255, 171, 84),
}
BACKGROUND = (14, 17, 24)
GRID = (48, 55, 70)
TEXT = (235, 240, 248)
MUTED = (150, 160, 178)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gymnasium + pygame 테트리스 정책 시뮬레이션")
    parser.add_argument("--model", type=Path, default=None, help="시각화할 모델 경로")
    parser.add_argument("--policy", choices=["model", "heuristic"], default="model", help="사용할 정책")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2], help="환경 stage")
    parser.add_argument("--episodes", type=int, default=1, help="재생할 에피소드 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--max-steps", type=int, default=500, help="에피소드 최대 스텝 수")
    parser.add_argument("--delay", type=float, default=0.08, help="스텝 사이 대기 시간")
    parser.add_argument("--cell-size", type=int, default=30, help="보드 셀 크기")
    parser.add_argument("--auto-close", action="store_true", help="마지막 에피소드 종료 후 창을 자동으로 닫습니다")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "tetris_rl" / "models", help="기본 모델 조회 경로")
    return parser.parse_args()


def get_font(size: int) -> pygame.font.Font:
    font = pygame.font.SysFont("malgungothic", size)
    if font is None:
        font = pygame.font.SysFont(None, size)
    return font


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, x: int, y: int, color: tuple[int, int, int] = TEXT) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))


def draw_board(surface: pygame.Surface, env: TetrisEnv, cell_size: int, offset_x: int, offset_y: int) -> None:
    for row in range(env.height):
        for col in range(env.width):
            rect = pygame.Rect(offset_x + col * cell_size, offset_y + row * cell_size, cell_size, cell_size)
            color = BOARD_COLORS[int(env.board[row, col])]
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, GRID, rect, 1)


def draw_piece_preview(
    surface: pygame.Surface,
    block_name: str,
    x: int,
    y: int,
    cell_size: int,
) -> None:
    matrix = BLOCK_ROTATIONS[block_name][0]
    color = BLOCK_COLORS[block_name]
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col] == 0:
                continue
            rect = pygame.Rect(x + col * cell_size, y + row * cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, GRID, rect, 1)


def choose_action(obs: np.ndarray, env: TetrisEnv, policy_name: str, model: Any | None, uses_action_mask: bool) -> int:
    if policy_name == "heuristic":
        return choose_heuristic_action(obs, env)
    if model is None:
        raise ValueError("모델 정책을 사용하려면 모델이 필요합니다.")
    if uses_action_mask:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
    else:
        action, _ = model.predict(obs, deterministic=True)
    return int(np.asarray(action).item())


def run_viewer(args: argparse.Namespace) -> None:
    pygame.init()
    board_w = 10 * args.cell_size
    board_h = 20 * args.cell_size
    side_w = 260
    padding = 24
    screen = pygame.display.set_mode((board_w + side_w + padding * 3, board_h + padding * 2))
    pygame.display.set_caption("Tetris RL Simulation")
    clock = pygame.time.Clock()
    title_font = get_font(24)
    font = get_font(18)
    small_font = get_font(15)

    model = None
    uses_action_mask = False
    model_path: Path | None = None
    if args.policy == "model":
        model_path = args.model if args.model else find_default_model(args.model_dir)
        model, uses_action_mask = load_policy_model(model_path)
        print(f"모델을 불러왔습니다: {model_path}")
        if uses_action_mask:
            print("Action Masking을 사용해 시뮬레이션합니다.")
    else:
        print("휴리스틱 정책으로 시뮬레이션합니다.")

    print("조작: Space 일시정지/재개, N 한 스텝 진행, R 에피소드 재시작, Esc 또는 Q 종료")

    env = TetrisEnv(stage=args.stage, max_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    episode = 1
    total_reward = 0.0
    paused = False
    step_once = False
    running = True
    done = False
    last_info = {"steps": 0, "lines_cleared": 0}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    step_once = True
                elif event.key == pygame.K_r:
                    obs, _ = env.reset(seed=args.seed + episode)
                    total_reward = 0.0
                    done = False
                    last_info = {"steps": 0, "lines_cleared": 0}

        should_step = running and not done and (not paused or step_once)
        if should_step:
            action = choose_action(obs, env, args.policy, model, uses_action_mask)
            obs, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            step_once = False

            if done:
                print(
                    f"에피소드 {episode} 종료: 생존 {last_info['steps']}스텝, "
                    f"라인 {last_info['lines_cleared']}개, 보상 {total_reward:.2f}"
                )
                if episode >= args.episodes:
                    if args.auto_close:
                        running = False
                    else:
                        paused = True
                else:
                    episode += 1
                    obs, _ = env.reset(seed=args.seed + episode)
                    total_reward = 0.0
                    done = False
                    last_info = {"steps": 0, "lines_cleared": 0}

        screen.fill(BACKGROUND)
        draw_board(screen, env, args.cell_size, padding, padding)

        side_x = padding * 2 + board_w
        draw_text(screen, title_font, "Tetris RL", side_x, padding, TEXT)
        draw_text(screen, font, f"정책: {args.policy}", side_x, padding + 42, MUTED)
        draw_text(screen, font, f"Stage: {env.stage}", side_x, padding + 72, TEXT)
        draw_text(screen, font, f"Episode: {episode}/{args.episodes}", side_x, padding + 102, TEXT)
        draw_text(screen, font, f"Steps: {last_info['steps']}", side_x, padding + 132, TEXT)
        draw_text(screen, font, f"Lines: {last_info['lines_cleared']}", side_x, padding + 162, TEXT)
        draw_text(screen, font, f"Reward: {total_reward:.1f}", side_x, padding + 192, TEXT)
        draw_text(screen, font, f"Block: {env.current_block}", side_x, padding + 232, TEXT)
        draw_piece_preview(screen, env.current_block, side_x, padding + 262, max(16, args.cell_size // 2))
        draw_text(screen, font, f"Next: {env.next_block}", side_x, padding + 342, TEXT)
        draw_piece_preview(screen, env.next_block, side_x, padding + 372, max(16, args.cell_size // 2))

        if paused:
            draw_text(screen, small_font, "일시정지", side_x, board_h + padding - 58, (255, 214, 102))
        draw_text(screen, small_font, "Space/N/R/Q", side_x, board_h + padding - 30, MUTED)
        if model_path is not None:
            draw_text(screen, small_font, model_path.name[:26], side_x, board_h + padding - 86, MUTED)

        pygame.display.flip()
        clock.tick(60)
        if should_step and args.delay > 0:
            time.sleep(args.delay)

    env.close()
    pygame.quit()
    print("시뮬레이션을 종료했습니다.")


def main() -> None:
    args = parse_args()
    run_viewer(args)


if __name__ == "__main__":
    main()
