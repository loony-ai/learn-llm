"""The smallest model that predicts at every position: a bigram model (Chapter 11.8).

Each position's prediction depends only on the token at that position: an
embedding table whose rows are read directly as logits for the next token.
It is Chapter 1's one-word counting model, learned by gradient descent instead
of counted, with exactly the input/output shapes a transformer will have:

    token_ids (batch, seq) -> logits (batch, seq, vocab_size)
"""

from __future__ import annotations

import torch
from torch import nn


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)   # row t = logits for whatever follows token t

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.table(token_ids)
