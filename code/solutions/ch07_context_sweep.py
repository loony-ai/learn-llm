"""Chapter 7, Exercise 1 (suggested solution): counting versus network across context sizes.

Run from `code/`:  python -m solutions.ch07_context_sweep
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from llmfp.char_model import CharMLP, CharModelConfig, CharVocabulary, counting_baseline, make_examples
from llmfp.counting_lm import read_lines
from llmfp.experiment import set_seed
from llmfp.splits import deduplicate, hash_split
from llmfp.training_basics import fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8, 16, 24])
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    splits = hash_split(deduplicate(read_lines("data/tiny/harbor_synth.txt")), salt="0")
    train_text, val_text = "\n".join(splits.train) + "\n", "\n".join(splits.validation) + "\n"
    vocabulary = CharVocabulary.build(train_text)
    print(f"{'context':>7} | {'counting acc':>12} | {'coverage':>8} | {'network acc':>11} | {'network on unseen':>17}")
    for size in args.sizes:
        set_seed(0)
        train = make_examples(vocabulary.encode(train_text), size)
        val = make_examples(vocabulary.encode(val_text), size)
        counting = counting_baseline(*train, *val)
        model = CharMLP(CharModelConfig(vocabulary.size, size, 128))
        history = fit(model, train, val, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), lr=0.003), args.epochs, 128)
        model.eval()
        with torch.no_grad():
            correct = model(val[0]).argmax(dim=-1) == val[1]
        unseen = ~counting["seen"]
        unseen_text = f"{correct[unseen].float().mean():.1%} of {int(unseen.sum())}" if unseen.any() else "none unseen"
        print(f"{size:>7} | {counting['accuracy']:>12.1%} | {counting['coverage']:>8.1%} | "
              f"{history[-1]['val_accuracy']:>11.1%} | {unseen_text:>17}")


if __name__ == "__main__":
    main()
