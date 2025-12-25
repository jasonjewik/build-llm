from argparse import ArgumentParser

from src import the_verdict
from src.attention import (
    CausalAttention,
    SelfAttentionV1,
    SelfAttentionV2,
    SimpleSelfAttention,
)
from src.dataloader import create_dataloader_v1
from src.embedding import SimpleEmbeddingV1


def main():
    parser = ArgumentParser(
        description="Showcases different attention mechanisms.",
    )
    parser.add_argument(
        "version",
        choices=["simple", "v1", "v2", "causal"],
    )
    args = parser.parse_args()

    raw_text = the_verdict.get()
    max_length = 6  # the maximum length of any sample
    # Only the causal attention mechanism is implemented to accept a batch of
    # inputs (as opposed to a single sample).
    batch_size = 2 if args.version == "causal" else 1
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
        case "simple":
            self_attention = SimpleSelfAttention(context_length=max_length)
        case "v1":
            self_attention = SelfAttentionV1(
                d_in=embedding_dim,
                # Pick d_out != d_in for ease of visualization.
                d_out=2,
            )
        case "v2":
            self_attention = SelfAttentionV2(
                d_in=embedding_dim,
                d_out=2,
                qkv_bias=False,
            )
        case "causal":
            self_attention = CausalAttention(
                d_in=embedding_dim,
                d_out=2,
                context_length=max_length,
                dropout=0.2,
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
