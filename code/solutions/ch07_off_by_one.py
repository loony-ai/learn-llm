"""Chapter 7, Exercise 4 (suggested solution): the off-by-one target bug.

The correct target for a window is the character AFTER it. Here the targets are
taken one position too early: each target is the LAST character of its own window.
The task becomes "copy the last character you were given", which is trivial, so
the loss collapses and accuracy looks perfect, while generation is useless.

Run from `code/`:  python -m solutions.ch07_off_by_one
"""

from __future__ import annotations

import torch
from torch import nn

from llmfp.char_model import CharMLP, CharModelConfig, CharVocabulary, sample_text
from llmfp.counting_lm import read_lines
from llmfp.experiment import set_seed
from llmfp.splits import deduplicate, hash_split
from llmfp.training_basics import fit


def make_examples_with_bug(ids: list[int], context_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.tensor(ids, dtype=torch.int64)
    contexts = sequence.unfold(0, context_size, 1)[:-1]
    targets = sequence[context_size - 1 : -1]          # BUG: should be sequence[context_size:]
    return contexts.contiguous(), targets


def main() -> None:
    set_seed(0)
    splits = hash_split(deduplicate(read_lines("data/tiny/harbor_synth.txt")), salt="0")
    train_text, val_text = "\n".join(splits.train) + "\n", "\n".join(splits.validation) + "\n"
    vocabulary = CharVocabulary.build(train_text)
    train = make_examples_with_bug(vocabulary.encode(train_text), 12)
    val = make_examples_with_bug(vocabulary.encode(val_text), 12)
    model = CharMLP(CharModelConfig(vocabulary.size, 12, 128))
    history = fit(model, train, val, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), lr=0.003), 2, 128)
    for record in history:
        print(f"epoch {record['epoch']}: train loss {record['train_loss']:.4f}, val loss {record['val_loss']:.4f}, val acc {record['val_accuracy']:.1%}")
    print("Greedy:", repr(sample_text(model, vocabulary, "The keeper ", 40, greedy=True)))


if __name__ == "__main__":
    main()
