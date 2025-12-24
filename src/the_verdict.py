import os

import requests


def get() -> str:
    """Returns the text of the-verdict.txt, downloading first if not available locally."""
    target_file = "the-verdict.txt"
    if os.path.isfile(target_file):
        with open("the-verdict.txt", mode="r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        with requests.get(
            f"https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/{target_file}",
        ) as resp:
            if not resp.ok:
                raise RuntimeError(
                    f"Could not retrieve target file: status code {resp.status_code}"
                )
            raw_text = resp.text
        with open(target_file, mode="w", encoding="utf-8") as f:
            f.write(raw_text)
    return raw_text
