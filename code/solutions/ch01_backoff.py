"""Chapter 1, Exercise 5 (suggested solution): a backoff counting model.

When the full context was never seen, fall back to a shorter context, then a
shorter one, down to a single word. This was a standard trick in counting-based
language models; neural models (Chapter 5 onward) address the same problem of
unseen contexts in a different way.

Run from `code/`:
    python -m solutions.ch01_backoff --prompt "the gulls lit"
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import Iterable

from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig, read_lines


class BackoffLanguageModel(CountingLanguageModel):
    """A counting model that tries shorter contexts when the longest one is unseen."""

    def __init__(self, config: CountingModelConfig) -> None:
        super().__init__(config)
        # One plain counting model per shorter context size, longest first.
        self.shorter_models = [
            CountingLanguageModel(CountingModelConfig(context_size=size, lowercase=config.lowercase))
            for size in range(config.context_size - 1, 0, -1)
        ]
        self.last_context_size_used: int | None = None

    def train(self, lines: Iterable[str]) -> int:
        lines = list(lines)  # we iterate several times, so a one-shot iterator would break
        for model in self.shorter_models:
            model.train(lines)
        return super().train(lines)

    def followers_for(self, words: list[str]) -> Counter[str] | None:
        # Call the parent-class lookup directly on each model, longest context first.
        for model in [self, *self.shorter_models]:
            followers = CountingLanguageModel.followers_for(model, words)
            if followers is not None:
                self.last_context_size_used = model.config.context_size
                return followers
        self.last_context_size_used = None
        return None

    def num_parameters(self) -> int:
        return super().num_parameters() + sum(m.num_parameters() for m in self.shorter_models)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare plain and backoff counting models")
    parser.add_argument("--data", default="data/tiny/harbor.txt")
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--prompt", default="the gulls lit")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lines = read_lines(args.data)
    config = CountingModelConfig(context_size=args.context_size)
    plain = CountingLanguageModel(config)
    plain.train(lines)
    backoff = BackoffLanguageModel(config)
    backoff.train(lines)

    print(f"Prompt: {args.prompt!r}   context_size={args.context_size}")
    print(f"Plain model candidates:   {plain.next_word_candidates(args.prompt)}")
    print(f"Backoff model candidates: {backoff.next_word_candidates(args.prompt)}")
    print(f"  (backoff used a context of {backoff.last_context_size_used} word(s))")
    for name, model in [("plain", plain), ("backoff", backoff)]:
        result = model.generate(args.prompt, max_new_words=12, rng=random.Random(args.seed))
        print(f"{name:>8} generate: {result.text!r} [stopped: {result.stop_reason}]")


if __name__ == "__main__":
    main()
