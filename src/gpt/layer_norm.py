from typing import Self

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Compared to batch norm, which normalizes across the batch dimension, layer
    norm normalizes across the _feature_ dimension. Since each input is thus
    normalized independently of the batch size, it offers more flexibility and
    stability across different kinds of hardware or even distributed training
    (which affects batch size).
    """

    def __init__(self: Self, emb_dim: int) -> None:
        super().__init__()
        # Small constant added to the variance to prevent division by zero
        # during normalization.
        self.eps = 1e-5
        # Trainable parameters of the same dimension as the input that the LLM
        # can adjust during training if it is determined that doing so would
        # improve the model's performance on its training task.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
