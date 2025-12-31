from argparse import ArgumentParser
from enum import StrEnum

from src import the_verdict
from src.attention import (
    CausalAttention,
    MultiHeadAttentionV1,
    MultiHeadAttentionV2,
    SelfAttentionV1,
    SelfAttentionV2,
    SimpleSelfAttention,
)
from src.dataloader import create_dataloader_v1
from src.embedding import SimpleEmbeddingV1


class Attention(StrEnum):
    SIMPLE = "simple"
    V1 = "v1"
    V2 = "v2"
    CAUSAL = "causal"
    MULTI_HEAD_V1 = "multi-head-v1"
    MULTI_HEAD_V2 = "multi-head-v2"


def main():
    parser = ArgumentParser(
        description="Showcases different attention mechanisms.",
    )
    parser.add_argument(
        "version",
        choices=[
            Attention.SIMPLE,
            Attention.V1,
            Attention.V2,
            Attention.CAUSAL,
            Attention.MULTI_HEAD_V1,
            Attention.MULTI_HEAD_V2,
        ],
    )
    args = parser.parse_args()

    raw_text = the_verdict.get()
    max_length = 6  # the maximum length of any sample
    # The causal and multihead attention mechanisms are implemented to accept a
    # batch of inputs (as opposed to a single sample).
    batch_size = (
        2
        if args.version
        in {
            Attention.CAUSAL,
            Attention.MULTI_HEAD_V1,
            Attention.MULTI_HEAD_V2,
        }
        else 1
    )
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,  # stride = max_length means no overlapping windows
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, _ = next(data_iter)

    # Pick a small embedding dimension for ease of visualization.
    embedding_dim = 3
    embedding_model = SimpleEmbeddingV1(
        context_length=max_length,
        output_dim=embedding_dim,
    )
    input_embedding = embedding_model.forward(inputs)

    match args.version:
        case Attention.SIMPLE:
            self_attention = SimpleSelfAttention(context_length=max_length)
        case Attention.V1:
            self_attention = SelfAttentionV1(
                d_in=embedding_dim,
                # Pick d_out != d_in for ease of visualization.
                d_out=2,
            )
        case Attention.V2:
            self_attention = SelfAttentionV2(
                d_in=embedding_dim,
                d_out=2,
                qkv_bias=False,
            )
        case Attention.CAUSAL:
            self_attention = CausalAttention(
                d_in=embedding_dim,
                d_out=2,
                context_length=max_length,
                dropout=0.2,
                qkv_bias=False,
            )
        case Attention.MULTI_HEAD_V1:
            self_attention = MultiHeadAttentionV1(
                d_in=embedding_dim,
                d_out=1,  # out dim of 1...
                context_length=max_length,
                dropout=0.2,
                num_heads=2,  # ... with 2 heads means effective d_out=2
                qkv_bias=False,
            )
        case Attention.MULTI_HEAD_V2:
            self_attention = MultiHeadAttentionV2(
                d_in=embedding_dim,
                d_out=2,  # no need to adjust d_out
                context_length=max_length,
                dropout=0.2,
                num_heads=2,
                qkv_bias=False,
            )
        case _:
            raise ValueError(
                f"Invalid attention version specified: {args.version}",
            )

    if batch_size == 1:
        inputs = input_embedding[0]
    else:
        inputs = input_embedding
    context_vectors = self_attention.forward(inputs)
    print("Context vectors:\n", context_vectors)


if __name__ == "__main__":
    main()
