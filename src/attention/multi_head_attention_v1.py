from typing import Self

import torch
import torch.nn as nn

from src.attention.causal_attention import CausalAttention


class MultiHeadAttentionV1(nn.Module):
    def __init__(
        self: Self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        context_matrix = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        if self.verbose:
            print(f"{context_matrix.shape=}")
        return context_matrix
