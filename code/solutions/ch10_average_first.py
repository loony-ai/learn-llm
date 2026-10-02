"""Chapter 10, Exercise 5 (suggested solution): the "average first" mistake, measured.

A variant of the bag_position model that averages (token + position) embeddings
BEFORE any nonlinear layer. The average of positions is the same for every
context, so it adds a constant and carries no order information.

Run from `code/`:  python -m solutions.ch10_average_first
"""

from __future__ import annotations

import torch
from torch import nn

from llmfp.char_model import make_examples
from llmfp.counting_lm import read_lines
from llmfp.experiment import set_seed
from llmfp.model.embedding_mlp import EmbeddingMLP, EmbeddingMLPConfig
from llmfp.splits import deduplicate, hash_split
from llmfp.tokenizers import load_tokenizer
from llmfp.training_basics import fit


class AverageFirstModel(EmbeddingMLP):
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        averaged = self.embed(contexts).mean(dim=1)          # (batch, d_model): averaged FIRST
        return self.output(torch.relu(self.hidden(averaged)))


def main() -> None:
    tokenizer = load_tokenizer("data/tokenizer/harbor-bpe-2048.json")
    splits = hash_split(deduplicate(read_lines("data/tiny/harbor_synth.txt")), salt="0")
    train = make_examples(tokenizer.encode("\n".join(splits.train) + "\n"), 8)
    val = make_examples(tokenizer.encode("\n".join(splits.validation) + "\n"), 8)
    for label, cls, mode in [("bag (no positions)", EmbeddingMLP, "bag"),
                             ("positions, averaged first", AverageFirstModel, "bag_position"),
                             ("positions, layer before averaging", EmbeddingMLP, "bag_position")]:
        set_seed(0)
        model = cls(EmbeddingMLPConfig(tokenizer.vocab_size, 8, 32, 128, mode))
        history = fit(model, train, val, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), lr=0.003), 10, 128)
        print(f"{label:<34} val loss {history[-1]['val_loss']:.4f}, val accuracy {history[-1]['val_accuracy']:.1%}")


if __name__ == "__main__":
    main()
