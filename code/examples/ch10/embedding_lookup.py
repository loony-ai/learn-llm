"""Chapter 10.3: an embedding table is a learned lookup, equivalent to one-hot times a table.

Run from `code/`:  python examples/ch10/embedding_lookup.py
"""

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
table = nn.Embedding(num_embeddings=6, embedding_dim=3)     # 6 tokens, 3 numbers each
print("table.weight shape:", tuple(table.weight.shape), "-> one row per token ID")

ids = torch.tensor([[4, 1, 4]])                             # (batch=1, seq=3)
vectors = table(ids)
print("ids shape", tuple(ids.shape), "-> vectors shape", tuple(vectors.shape))
print("row for ID 4 appears twice:", torch.equal(vectors[0, 0], vectors[0, 2]), "and equals table row 4:",
      torch.equal(vectors[0, 0], table.weight[4]))

# The same result via Chapter 7's one-hot route: one-hot (1, 3, 6) times the table (6, 3).
one_hot = F.one_hot(ids, num_classes=6).float()
print("one-hot @ table gives the same vectors:", torch.allclose(one_hot @ table.weight, vectors))

# Training updates only the rows that were looked up.
vectors.sum().backward()
used = [i for i in range(6) if table.weight.grad[i].abs().sum() > 0]
print("rows that received a gradient:", used, "(IDs 1 and 4; the rest are untouched)")

# An ID outside the table is an error, a common symptom of a tokenizer/model mismatch.
try:
    table(torch.tensor([6]))
except IndexError as error:
    print("ID 6 in a 6-row table ->", type(error).__name__ + ":", error)
