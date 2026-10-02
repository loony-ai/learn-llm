"""Chapter 9, Exercise 3 (suggested solution): how compression changes with vocabulary size.

Trains BPE tokenizers of several sizes on the Project 1 corpus and measures
characters per token on held-out text (not in the training corpus).

Run from `code/`:  python -m solutions.ch09_vocab_sweep
"""

from __future__ import annotations

import argparse
import time

from llmfp.tokenizers import BPETokenizer
from scripts.ch09_compare_tokenizers import held_out_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    args = parser.parse_args()

    corpus = open("data/tokenizer/corpus.txt", encoding="utf-8").read()
    texts = held_out_texts()
    short = {"harbor (new seed)": "harbor", "English prose (Ch 8 source)": "prose", "Python code (Ch 9 script)": "code",
             "multilingual sample": "multiling."}
    print(f"{'vocab':>6} {'merges':>7} {'train s':>8}" + "".join(f"{short[k]:>11}" for k in texts))
    for size in args.sizes:
        start = time.perf_counter()
        tokenizer = BPETokenizer.train(corpus, size)
        seconds = time.perf_counter() - start
        row = f"{tokenizer.vocab_size:>6} {len(tokenizer.merges):>7} {seconds:>8.1f}"
        for text in texts.values():
            row += f"{len(text) / len(tokenizer.encode(text)):>11.2f}"
        print(row)


if __name__ == "__main__":
    main()
