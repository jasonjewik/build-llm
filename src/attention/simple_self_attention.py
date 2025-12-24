from typing import Self

import torch


class SimpleSelfAttention(torch.nn.Module):
    def __init__(self: Self, context_length: int) -> None:
        self.context_length = context_length

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2:
            raise ValueError(
                "Expected input of shape context_length x embedding_dim ",
                f"got {len(x.shape)=}",
            )
        if x.shape[0] != self.context_length:
            raise ValueError(
                f"Expected input context length {self.context_length} "
                f"but got context length {x.shape[1]}",
            )
        # We compute an attention score for each token against every other
        # token (including itself) as the dot product.
        attn_scores = x @ x.T
        print("Attention scores:\n", attn_scores)
        # Normalize the scores across the columns so that each row sums to 1,
        # useful for interpretation and training stability. Think of each row
        # as representing a token so that a row of [0.9, 0.1, ...] means the
        # first word is the most important, the second is less so, etc.
        attn_weights = torch.softmax(attn_scores, dim=1)
        print("Normalized attention scores:\n", attn_weights)
        print("All row sums:", attn_weights.sum(dim=1))
        # Then we use the attention weights to compute a weighted sum of the
        # input vectors.
        context_vectors = attn_weights @ x
        return context_vectors
