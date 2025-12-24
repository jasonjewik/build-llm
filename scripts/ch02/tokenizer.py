from argparse import ArgumentParser
from enum import StrEnum
import re

import tiktoken

from src import the_verdict
from src.tokenizer import SimpleTokenizer, SimpleTokenizerV1, SimpleTokenizerV2


class SpecialToken(StrEnum):
    END_OF_TEXT = "<|endoftext|>"
    UNKNOWN = "<|unk|>"
    # Depending on the LLM, some additional special tokens might be used
    # such as [BOS], [EOS], [PAD].


def main():
    parser = ArgumentParser(
        description="Showcases the capabilities of different tokenizers.",
    )
    parser.add_argument("tokenizer", choices=["v1", "v2", "bpe"])
    args = parser.parse_args()

    raw_text = the_verdict.get()
    vocabulary = extract_vocabulary(raw_text)

    match args.tokenizer:
        case "v1":
            tokenizer = SimpleTokenizerV1(vocabulary=vocabulary)
        case "v2":
            vocabulary[SpecialToken.END_OF_TEXT] = len(vocabulary)
            vocabulary[SpecialToken.UNKNOWN] = len(vocabulary)
            tokenizer = SimpleTokenizerV2(vocabulary=vocabulary)
        case "bpe":
            tokenizer = tiktoken.get_encoding("gpt2")
        case _:
            raise ValueError("Invalid tokenizer selected")

    test_strings = [
        # Both tokenizers should succeed here.
        "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.",
        # SimpleTokenizerV1 fails since "Hello" is not in its vocabulary.
        # SimpleTokenizerV2 should replace "Hello" with <|unk|>.
        "Hello, do you like tea?",
        # SimpleTokenizerV1 never makes it here (fails on the previous test string).
        # SimpleTokenizerV2 should replace "Hello" and "palace" with <|unk|>.
        # It should preserve <|endoftext|>.
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.",
    ]

    for s in test_strings:
        try:
            run_tokenizer(tokenizer, text=s)
        except Exception as e:
            print(f"Could not tokenize input: {str(e)}")
            break


def extract_vocabulary(raw_text: str) -> dict[str, int]:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    return {token: integer for integer, token in enumerate(all_words)}


def run_tokenizer(
    tokenizer: SimpleTokenizer | tiktoken.Encoding,
    text: str,
) -> None:
    print("Input text:", text)
    if isinstance(tokenizer, tiktoken.Encoding):
        ids = tokenizer.encode(text, allowed_special={SpecialToken.END_OF_TEXT})
    else:
        ids = tokenizer.encode(text)
    print("IDs:", ids)
    decoded_text = tokenizer.decode(ids)
    print("Decoded text:", decoded_text, "\n")


if __name__ == "__main__":
    main()
