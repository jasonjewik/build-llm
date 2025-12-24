from src import the_verdict
from src.dataloader import create_dataloader_v1
from src.embedding import SimpleEmbeddingV1


def main():
    raw_text = the_verdict.get()
    max_length = 4  # the maximum length of any sample in tokens
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,  # stride = max_length means no overlapping windows
        shuffle=False, 
    )
    data_iter = iter(dataloader)
    inputs, _ = next(data_iter)
    embedding_model = SimpleEmbeddingV1(context_length=max_length)
    input_embedding = embedding_model.forward(inputs)
    print("Input embedding:\n", input_embedding)


if __name__ == "__main__":
    main()