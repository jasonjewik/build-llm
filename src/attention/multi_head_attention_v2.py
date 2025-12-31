from typing import Self

import torch
import torch.nn as nn


class MultiHeadAttentionV2(nn.Module):
    """
    More efficient implementation of multi-head attention with parallel
    processing of the attention heads. Further, unlike MultiHeadAttentionV1,
    which merely wraps CausalAttention, this implementation folds in the causal
    attention mechanism.
    """

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
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        # Reduces the projection dim to match the desired output dim
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer(
            name="mask",
            tensor=torch.triu(
                input=torch.ones(context_length, context_length),
                diagonal=1,
            ),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        keys = self.W_key.forward(x)
        queries = self.W_query.forward(x)
        values = self.W_value.forward(x)
        if self.verbose:
            print(f"Original KQV shape: {keys.shape}")
        # We implicitly split the matrix by adding a num_heads dimension. Then
        # we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens,
        # num_heads, head_dim).
        split_shape = (b, num_tokens, self.num_heads, self.head_dim)
        if self.verbose:
            print(f"Reshaped to: {split_shape}")
        keys = keys.view(*split_shape)
        values = values.view(*split_shape)
        queries = queries.view(*split_shape)
        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b,
        # num_heads, num_tokens, head_dim).
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        if self.verbose:
            print(f"Transposed to: {keys.shape}")
        # Compute the dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        if self.verbose:
            print("Raw attention scores:\n", attn_scores)
        # Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore
        # Use the mask to fill atention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout.forward(attn_weights)
        if self.verbose:
            print("Attention weights:\n", attn_weights)
        # Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        if self.verbose:
            print(f"Original context vector shape: {context_vec.shape}")
        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        if self.verbose:
            print(f"Combined context vector shape: {context_vec.shape}")
        # Adds an optional linear projection
        context_vec = self.out_proj.forward(context_vec)
        return context_vec
