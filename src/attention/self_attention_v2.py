from typing import Self

import torch
import torch.nn as nn


class SelfAttentionV2(nn.Module):
    """Using nn.Linear for optimized weight initialization."""

    def __init__(
        self: Self,
        d_in: int,
        d_out: int,
        qkv_bias: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2:
            raise ValueError(
                "Expected input of shape context_length x embedding_dim ",
                f"got {len(x.shape)=}",
            )
        keys = self.W_key.forward(x)
        queries = self.W_query.forward(x)
        values = self.W_value.forward(x)
        if self.verbose:
            print(f"{keys.shape=}")
            print(f"{queries.shape=}")
            print(f"{values.shape=}")
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[1] ** 0.5, dim=1)
        if self.verbose:
            print("Attention weights:\n", attn_weights)
        context_vector = attn_weights @ values
        return context_vector
