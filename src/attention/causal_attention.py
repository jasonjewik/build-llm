from typing import Self

import torch


class CausalAttention(torch.nn.Module):
    """
    Scaled-dot product attention where only previous tokens are considered for
    calculations on the current token. We also add a dropout layer.
    """

    def __init__(
        self: Self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(p=dropout)
        # Buffers are automatically moved to the appropriate device along with
        # the model.
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        keys = self.W_key.forward(x)
        queries = self.W_query.forward(x)
        values = self.W_value.forward(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # In PyTorch, operations with a trailing underscore are performed
        # in-place, avoiding unnecessary memory copies.
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],  # type: ignore
            -torch.inf,
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1,
        )
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values
        return context_vector
