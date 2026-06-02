from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TetrisFeatureExtractor(BaseFeaturesExtractor):
    """테트리스 상태 벡터를 구조별로 나눠 처리하는 PPO feature extractor입니다."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 192) -> None:
        super().__init__(observation_space, features_dim)

        self.height_net = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.hole_net = nn.Sequential(
            nn.Linear(10, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
        )
        self.block_net = nn.Sequential(
            nn.Linear(14, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
        )
        self.global_net = nn.Sequential(
            nn.Linear(7, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
        )
        self.final_net = nn.Sequential(
            nn.Linear(64 + 48 + 48 + 48, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        heights = observations[:, 0:10] / 20.0
        holes = observations[:, 10:20] / 20.0
        blocks = observations[:, 20:34]
        stage = observations[:, 34:35] / 2.0

        height_diffs = torch.abs(heights[:, 1:] - heights[:, :-1])
        aggregate_height = torch.sum(heights, dim=1, keepdim=True)
        total_holes = torch.sum(holes, dim=1, keepdim=True)
        bumpiness = torch.sum(height_diffs, dim=1, keepdim=True)
        max_height = torch.max(heights, dim=1, keepdim=True).values
        min_height = torch.min(heights, dim=1, keepdim=True).values
        height_span = max_height - min_height
        global_features = torch.cat(
            [aggregate_height, total_holes, bumpiness, max_height, min_height, height_span, stage],
            dim=1,
        )

        encoded = torch.cat(
            [
                self.height_net(heights),
                self.hole_net(holes),
                self.block_net(blocks),
                self.global_net(global_features),
            ],
            dim=1,
        )
        return self.final_net(encoded)


def get_tetris_policy_kwargs() -> dict:
    """Stable-Baselines3 PPO에 전달할 테트리스 전용 policy 설정입니다."""
    return {
        "features_extractor_class": TetrisFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "net_arch": {"pi": [192, 96], "vf": [192, 96]},
        "activation_fn": nn.ReLU,
    }
