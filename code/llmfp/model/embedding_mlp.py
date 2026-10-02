"""A next-token model for studying embeddings and word order (Chapter 10).

Three ways to combine a context of token embeddings before predicting:

    "concat"         place the embeddings side by side, then the hidden layer
                     (order is kept by each embedding's place in the list)
    "bag"            pass each embedding through the hidden layer on its own, then average
                     (order is lost: a "bag of tokens")
    "bag_position"   add a learned position embedding to each token first, then as "bag"

Why the hidden layer comes BEFORE averaging in the bag modes: averaging
"token + position" directly would equal (average of tokens) + (average of
positions), and the second part is the same for every context, so the order
information would cancel out. Exercise 10.5 explores that mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from llmfp.model.embeddings import TokenAndPositionEmbedding

MODES = ("concat", "bag", "bag_position")


@dataclass(frozen=True)
class EmbeddingMLPConfig:
    vocab_size: int
    context_size: int = 8
    d_model: int = 32
    hidden: int = 128
    mode: str = "concat"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


class EmbeddingMLP(nn.Module):
    def __init__(self, config: EmbeddingMLPConfig) -> None:
        super().__init__()
        self.config = config
        if config.mode == "bag_position":
            self.embed = TokenAndPositionEmbedding(config.vocab_size, config.context_size, config.d_model)
        else:
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
        hidden_inputs = config.d_model * (config.context_size if config.mode == "concat" else 1)
        self.hidden = nn.Linear(hidden_inputs, config.hidden)
        self.output = nn.Linear(config.hidden, config.vocab_size)

    @property
    def token_table(self) -> torch.Tensor:
        """The learned token embedding table, (vocab_size, d_model)."""
        module = self.embed.token if self.config.mode == "bag_position" else self.embed
        return module.weight

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        vectors = self.embed(contexts)                                  # (batch, context, d_model)
        if self.config.mode == "concat":
            hidden = torch.relu(self.hidden(vectors.flatten(start_dim=1)))  # (batch, hidden)
        else:
            per_token = torch.relu(self.hidden(vectors))                 # (batch, context, hidden)
            hidden = per_token.mean(dim=1)                               # (batch, hidden): averaged
        return self.output(hidden)
