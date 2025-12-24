from argparse import ArgumentParser

from src import the_verdict
from src.attention import (
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
    parser.add_argument("version", choices=["simple", "v1", "v2"])
    args = parser.parse_args()

    raw_text = the_verdict.get()
    max_length = 6  # the maximum length of any sample
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=1,  # one batch for simplicity
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
        case _:
            raise ValueError(
                f"Invalid attention version specified: {args.version}",
            )

    # All the attention mechanisms are coded to take a single sample (i.e.,
    # non-batched input).
    context_vectors = self_attention.forward(input_embedding[0])
    print("Context vectors:\n", context_vectors)


if __name__ == "__main__":
    main()
