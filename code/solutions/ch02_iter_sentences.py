"""Chapter 2, Exercise 4 (suggested solution): read sentences lazily from several files.

Run from `code/`:
    python -m solutions.ch02_iter_sentences data/tiny/harbor.txt data/tiny/harbor.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Iterator

from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig


def iter_sentences(paths: Iterable[str | Path]) -> Iterator[str]:
    """Yield stripped, non-empty lines from each file in turn, one line at a time.

    Files are opened only when the consumer reaches them, and each is closed as
    soon as it is finished, so memory use does not depend on file size.
    """
    for path in paths:
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield line


def main() -> None:
    paths = sys.argv[1:] or ["data/tiny/harbor.txt"]
    model = CountingLanguageModel(CountingModelConfig())
    observations = model.train(iter_sentences(paths))  # train() accepts any iterable
    print(f"Read {len(paths)} file(s) lazily: {observations} observations, {model.num_parameters()} parameters")


if __name__ == "__main__":
    main()
