"""Chapter 7, Exercise 5 (suggested solution): how often is generated text fluent but false?

The synthetic corpus was generated from a fixed table of which actor performs
which actions (scripts/ch04_make_harbor_corpus.py). That table is the "world".
This script trains the Project 0 model briefly, generates sentences, and sorts
each into: well-formed and true to the world; well-formed but claiming an
actor-action pair the world never contains; or not well-formed.

Run from `code/`:  python -m solutions.ch07_fact_check
"""

from __future__ import annotations

import argparse
import re

import torch
from torch import nn

from llmfp.char_model import CharMLP, CharModelConfig, CharVocabulary, make_examples, sample_text
from llmfp.counting_lm import read_lines
from llmfp.experiment import set_seed
from llmfp.splits import deduplicate, hash_split
from llmfp.training_basics import fit
from scripts.ch04_make_harbor_corpus import ACTIONS, TIMES

ALL_ACTIONS = sorted({action for actions in ACTIONS.values() for action in actions}, key=len, reverse=True)
ACTORS = sorted(ACTIONS, key=len, reverse=True)
TIME_PATTERN = "|".join(re.escape(t) for t in TIMES)
ACTION_PATTERN = "|".join(re.escape(a) for a in ALL_ACTIONS)
ACTOR_PATTERN = "|".join(re.escape(a) for a in ACTORS)
SENTENCE = re.compile(
    rf"^(?:(?P<lead>{TIME_PATTERN}) )?(?P<actor>{ACTOR_PATTERN}) (?P<first>{ACTION_PATTERN})"
    rf"(?: and (?P<second>{ACTION_PATTERN}))?(?: (?P<tail>{TIME_PATTERN}))?\.$"
)


def classify(sentence: str) -> str:
    match = SENTENCE.match(sentence.strip().lower())
    if not match or bool(match["lead"]) == bool(match["tail"]):  # exactly one time phrase
        return "not well-formed"
    allowed = ACTIONS[match["actor"]]
    actions = [match["first"]] + ([match["second"]] if match["second"] else [])
    if all(action in allowed for action in actions):
        return "well-formed and true"
    return "well-formed but false"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sentences", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    set_seed(0)
    splits = hash_split(deduplicate(read_lines("data/tiny/harbor_synth.txt")), salt="0")
    train_text, val_text = "\n".join(splits.train) + "\n", "\n".join(splits.validation) + "\n"
    vocabulary = CharVocabulary.build(train_text)
    model = CharMLP(CharModelConfig(vocabulary.size, 12, 128))
    fit(model, make_examples(vocabulary.encode(train_text), 12), make_examples(vocabulary.encode(val_text), 12),
        nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), lr=0.003), args.epochs, 128)

    # Generate long streams and cut them into sentences. The first line of each stream is
    # dropped: it was generated from a context of padding newlines never seen in training.
    # The last line is dropped because it is unfinished.
    generator = torch.Generator().manual_seed(0)
    sentences: list[str] = []
    while len(sentences) < args.sentences:
        text = sample_text(model, vocabulary, "", 600, generator)
        sentences += [line for line in text.split("\n")[1:-1] if line]
    sentences = sentences[: args.sentences]

    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for sentence in sentences:
        label = classify(sentence)
        counts[label] = counts.get(label, 0) + 1
        examples.setdefault(label, sentence)
    copies = sum(sentence in set(splits.train) for sentence in sentences)
    print(f"{len(sentences)} generated sentences; {copies} are exact copies of training sentences")
    for label in ("well-formed and true", "well-formed but false", "not well-formed"):
        print(f"  {label:<22} {counts.get(label, 0):>4}   e.g. {examples.get(label, '-')!r}")


if __name__ == "__main__":
    main()
