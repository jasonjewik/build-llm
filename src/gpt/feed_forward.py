from typing import Self

import torch
import torch.nn as nn

from src.gpt.config import GptConfig
from src.gpt.gelu import GELU


class FeedForward(nn.Module):
    def __init__(self: Self, config: GptConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x)
