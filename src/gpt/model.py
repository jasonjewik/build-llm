from typing import Self

import torch
import torch.nn as nn

from src.gpt.config import GptConfig
from src.gpt.layer_norm import LayerNorm
from src.gpt.transformer import TransformerBlock


class GptModel(nn.Module):
    def __init__(self: Self, config: GptConfig) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.emb_dim,
        )
        self.position_emb = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.emb_dim,
        )
        self.drop_emb = nn.Dropout(p=config.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNorm(emb_dim=config.emb_dim)
        self.out_head = nn.Linear(
            in_features=config.emb_dim,
            out_features=config.vocab_size,
            bias=False,
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        # Takes a batch of input token indices
        batch_size, seq_len = x.shape
        # Computes their embeddings
        token_embeddings = self.token_emb.forward(x)
        # Apply the positional embeddings
        position_embeddings = self.position_emb.forward(
            torch.arange(seq_len, device=x.device),
        )
        x = token_embeddings + position_embeddings
        # Pass through the rest of the model
        x = self.drop_emb.forward(x)
        x = self.transformer_blocks.forward(x)
        x = self.final_norm.forward(x)
        logits = self.out_head.forward(x)
        return logits
