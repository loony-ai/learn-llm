"""Token embeddings, learned position embeddings, and similarity helpers (Chapter 10).

An embedding table is a learned lookup: row i is the list of numbers that
represents token i. Looking up a row is equivalent to multiplying a one-hot
vector by the table (Chapter 7.4), but without building the one-hot vector.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TokenAndPositionEmbedding(nn.Module):
    """token IDs (batch, seq) -> vectors (batch, seq, d_model).

    Each position's vector is its token's embedding plus its position's embedding,
    so the same token at different positions gets different vectors.
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context_length, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence of {sequence_length} tokens is longer than the {self.context_length} positions this model has"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)   # 0, 1, ..., seq - 1
        return self.token(token_ids) + self.position(positions)              # (batch, seq, d) + (seq, d): broadcast


def cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between every pair of rows: 1 = same direction, 0 = unrelated, -1 = opposite."""
    unit = F.normalize(vectors, dim=-1)   # rescale each row to length 1, keeping its direction
    return unit @ unit.T


def nearest_neighbors(table: torch.Tensor, index: int, k: int = 5) -> list[tuple[int, float]]:
    """The k rows most similar (by cosine similarity) to row `index`, excluding itself."""
    similarities = F.cosine_similarity(table[index].unsqueeze(0), table, dim=-1)
    similarities[index] = -2.0   # below any possible similarity, so it is never chosen
    top = similarities.topk(k)
    return list(zip(top.indices.tolist(), top.values.tolist()))
