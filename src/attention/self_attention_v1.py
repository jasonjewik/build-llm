from typing import Self

import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    """AKA scaled dot-product attention."""

    def __init__(self: Self, d_in: int, d_out: int, verbose: bool = False) -> None:
        # d_in = the token embedding dimension
        # d_out = the "attention" embedding dimension
        # In practice, usually d_in = d_out, but we can make them distinct here
        # for clarity.
        super().__init__()
        self.verbose = verbose
        # From each token, we will compute a query, key, and value. These terms
        # were derived from information retrieval, so you can think of them
        # like this: When we multiply a token embedding by the query weight
        # matrix, we get a "query vector" which we check for similarity against
        # the "key vectors", to determine which tokens are most relevant to
        # the query. From there, we can grab the corresponding "value vectors".
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 2:
            raise ValueError(
                "Expected input of shape context_length x embedding_dim ",
                f"got {len(x.shape)=}",
            )
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        if self.verbose:
            print(f"{keys.shape=}")
            print(f"{queries.shape=}")
            print(f"{values.shape=}")
        # Compare this against the simple self attention, where we computed the
        # attention scores directly on the input token embeddings. By using
        # keys, queries, and values computed from learnable parameters, the
        # model might find a "smarter" way. Intuitively, the model should learn
        # keys and queries which correspond to some notion of "relevance"
        # between tokens while it should learn values that correspond to some
        # notion of "enhanced meaning" of each token ("enhanced" because the
        # token embeddings should already capture some sense of meaning).
        attn_scores = queries @ keys.T
        # We normalize by the embedding  dimension size to improve training
        # performance. Large dot products can result in very small gradients
        # during backprop due to the softmax function, which behaves like a
        # step function for large-dimensional inputs (in GPT-like LLMs, the
        # embedding dimension is typically 1000+). Hence, we normalize by
        # the embedding dimension, giving us self-attention's other name
        # "scaled-dot product attention".
        # See for yourself:
        # t = torch.rand((1000,)) * 100
        # print(torch.softmax(t, dim=-1))
        # print(torch.softmax(t/10, dim=-1))
        attn_weights = torch.softmax(attn_scores / keys.shape[1] ** 0.5, dim=1)
        if self.verbose:
            print("Attention weights:\n", attn_weights)
        context_vector = attn_weights @ values
        return context_vector
