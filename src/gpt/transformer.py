from typing import Self

import torch
import torch.nn as nn

from src.attention import MultiHeadAttentionV2
from src.gpt.config import GptConfig
from src.gpt.feed_forward import FeedForward
from src.gpt.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self: Self, config: GptConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttentionV2(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            num_heads=config.n_heads,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias,
        )
        self.ff = FeedForward(config=config)
        self.norm1 = LayerNorm(emb_dim=config.emb_dim)
        self.norm2 = LayerNorm(emb_dim=config.emb_dim)
        self.drop_shortcut = nn.Dropout(p=config.drop_rate)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1.forward(x)
        x = self.attention.forward(x)
        x = self.drop_shortcut.forward(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2.forward(x)
        x = self.ff.forward(x)
        x = self.drop_shortcut.forward(x)
        x = x + shortcut

        return x
