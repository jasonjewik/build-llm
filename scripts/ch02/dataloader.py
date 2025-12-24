from argparse import ArgumentParser, Namespace

from src import the_verdict
from src.dataloader import create_dataloader_v1


def main():
    parser = ArgumentParser(
        description="Loads the text into a dataloader and shows the inputs/targets of the first batch.",
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if args.max_length <= 0:
        raise ValueError("Max length must be a positive integer")
    if args.stride <= 0:
        raise ValueError("Stride must be a positive integer")

    raw_text = the_verdict.get()
    dataloader = create_dataloader_v1(
        text=raw_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter) # get the first batch
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)


if __name__ == "__main__":
    main()