"""Measure how well a counting model predicts held-out text (Chapter 4).

For every position in every sentence, we ask the model for its single best next
word and compare it with the word that actually came next. Three numbers result:

    coverage  share of positions whose context the model saw during training
    accuracy  share of positions where the top-ranked word was the actual next word
              (a position with an unseen context counts as wrong)
    accuracy_when_covered  accuracy over the covered positions only

All three are shares between 0 and 1. They are counts divided by counts, nothing more.
"""

from __future__ import annotations

from typing import Iterable

from llmfp.counting_lm import END, START, CountingLanguageModel, split_into_words


def evaluate_counting_model(model: CountingLanguageModel, lines: Iterable[str]) -> dict[str, float | int]:
    size = model.config.context_size
    positions = covered = correct = 0
    for line in lines:
        words = split_into_words(line, model.config.lowercase)
        if not words:
            continue
        padded = [START] * size + words + [END]
        for position in range(size, len(padded)):
            positions += 1
            followers = model.counts.get(tuple(padded[position - size : position]))
            if not followers:
                continue
            covered += 1
            # Same ranking rule as next_word_candidates: highest count, ties alphabetical.
            best = min(followers.items(), key=lambda item: (-item[1], item[0]))[0]
            if best == padded[position]:
                correct += 1
    if positions == 0:
        raise ValueError("no positions to evaluate")
    return {
        "positions": positions,
        "coverage": covered / positions,
        "accuracy": correct / positions,
        "accuracy_when_covered": correct / covered if covered else 0.0,
    }
