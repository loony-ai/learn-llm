"""Chapter 3.3: shape, dimensions, and axes, using token IDs and features.

Run from `code/`:  python examples/ch03/shapes_axes.py
"""

import torch

scalar = torch.tensor(7)                         # one number
vector = torch.tensor([5, 9, 2, 5, 7])           # one sequence of token IDs
matrix = torch.tensor([[5, 9, 2, 5, 7],          # a batch: 3 sequences of 5 token IDs
                       [5, 1, 4, 0, 3],
                       [8, 9, 2, 6, 7]])
for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    print(f"{name:<7} shape={tuple(t.shape)!s:<8} dimensions={t.ndim}  elements={t.numel()}")

# A 3-dimensional tensor: for every token in every sequence, 4 numbers describing it.
# This (batch, sequence, features) layout is the one a transformer works with.
features = torch.arange(3 * 5 * 4).reshape(3, 5, 4)
print("\nfeatures shape:", tuple(features.shape), "-> (batch=3, sequence=5, features=4)")
print("features[0] is sequence 0, shape", tuple(features[0].shape))
print("features[0, 1] is token 1 of sequence 0, shape", tuple(features[0, 1].shape), "->", features[0, 1].tolist())

# An axis is one dimension, numbered from 0. Operations "along an axis" combine
# the elements that differ only in that axis.
print("\nmatrix:\n", matrix)
print("sum along axis 0 (down the columns, one result per position):", matrix.sum(dim=0).tolist())
print("sum along axis 1 (across each row, one result per sequence):  ", matrix.sum(dim=1).tolist())
print("negative axis -1 means the last axis:", matrix.sum(dim=-1).tolist())
