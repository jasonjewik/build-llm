from typing import Self

import torch


class SimpleEmbeddingV1(torch.nn.Module):
    """A simple, non-trainable embedding model to show the fundamental idea."""
    
    def __init__(
        self: Self,
        vocab_size: int = 50257,  # vocab size of the BPE tokenizer
        context_length: int = 4,  # number of tokens in each sample
        output_dim: int = 256,  # embedding vector dimension corresponding to each token
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        # We use the absolute position approach (as opposed to relative position).
        self.pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        # x expected to be shape batch_size x context_length
        if len(x.shape) != 2:
            raise ValueError(
                "Expected input tensor of shape batch_size x context_length, ",
                f"instead got: {x.shape=}"
            )
        if x.shape[1] != self.context_length:
            # In practice, the input text can be longer than the supported context length,
            # in which case we have to truncate the text. But here, for simplicity, we
            # assume that the input text's length must match the context length.
            raise ValueError(
                f"Mismatch in context length: expected {self.context_length}, got {x.shape[1]}",
            )
        token_embeddings = self.token_embedding_layer.forward(x)
        print(f"{token_embeddings.shape=}")
        pos_embeddings = self.pos_embedding_layer.forward(torch.arange(self.context_length))
        print(f"{pos_embeddings.shape=}")
        input_embeddings = token_embeddings + pos_embeddings
        print(f"{input_embeddings.shape=}")
        return input_embeddings
