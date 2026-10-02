"""Chapter 1, Exercise 6 (suggested solution): measure memorization vs context size.

For each context size, generate many sentences from an empty prompt and report:
  - how many generated sentences are exact copies of a training sentence
  - how many distinct sentences were generated
  - what share of contexts have only one possible next word

Run from `code/`:
    python -m solutions.ch01_memorization
"""

from __future__ import annotations

import argparse
import random

from llmfp.counting_lm import (
    CountingLanguageModel,
    CountingModelConfig,
    join_words,
    read_lines,
    split_into_words,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/tiny/harbor.txt")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-context-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lines = read_lines(args.data)
    training_sentences = {join_words(split_into_words(line, True)) for line in lines}

    print(f"{args.samples} samples per setting, seed {args.seed}, data {args.data}")
    print(f"{'context':>7} | {'params':>6} | {'1-follower contexts':>19} | {'exact copies':>12} | {'distinct':>8}")
    for size in range(1, args.max_context_size + 1):
        model = CountingLanguageModel(CountingModelConfig(context_size=size))
        model.train(lines)
        single = sum(1 for followers in model.counts.values() if len(followers) == 1)
        rng = random.Random(args.seed)
        texts = [model.generate("", max_new_words=30, rng=rng).text for _ in range(args.samples)]
        copies = sum(text in training_sentences for text in texts)
        print(
            f"{size:>7} | {model.num_parameters():>6} | "
            f"{single:>4} of {model.num_contexts():<3} ({single / model.num_contexts():>4.0%}) | "
            f"{copies:>4} ({copies / args.samples:>4.0%}) | {len(set(texts)):>8}"
        )


if __name__ == "__main__":
    main()
