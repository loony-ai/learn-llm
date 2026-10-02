"""Chapter 11, Exercise 4 (suggested solution): reduce padding by batching similar lengths together.

Run from `code/`:  python -m solutions.ch11_bucketing
"""

from __future__ import annotations

import random

from llmfp.counting_lm import read_lines
from llmfp.data import pad_batch
from llmfp.splits import deduplicate
from llmfp.tokenizers import load_tokenizer


def padding_waste(batches: list[list[list[int]]]) -> float:
    """Share of positions that are padding, over all batches."""
    total = real = 0
    for batch in batches:
        padded = pad_batch(batch, pad_id=0)
        total += padded.attention_mask.numel()
        real += int(padded.attention_mask.sum())
    return 1 - real / total


def random_batches(documents: list[list[int]], batch_size: int, seed: int) -> list[list[list[int]]]:
    shuffled = documents[:]
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def bucketed_batches(documents: list[list[int]], batch_size: int, seed: int) -> list[list[list[int]]]:
    """Sort by length, cut into batches, then shuffle the ORDER of batches (not their contents)."""
    ordered = sorted(documents, key=len)
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    random.Random(seed).shuffle(batches)
    return batches


def main() -> None:
    tokenizer = load_tokenizer("data/tokenizer/harbor-bpe-2048.json")
    documents = [tokenizer.encode(line) for line in deduplicate(read_lines("data/tiny/harbor_synth.txt"))]
    for batch_size in (8, 32, 128):
        print(f"batch size {batch_size:>3}: padding waste random {padding_waste(random_batches(documents, batch_size, 0)):.1%}, "
              f"bucketed {padding_waste(bucketed_batches(documents, batch_size, 0)):.1%}")


if __name__ == "__main__":
    main()
