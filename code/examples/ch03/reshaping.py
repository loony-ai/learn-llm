"""Chapter 3.6: reshaping, views, transposes, and contiguity.

Run from `code/`:  python examples/ch03/reshaping.py
"""

import torch

t = torch.arange(12)
print("original      ", tuple(t.shape), t.tolist())
print("reshape(3, 4) \n", t.reshape(3, 4))
print("reshape(2, -1)", tuple(t.reshape(2, -1).shape), "(-1 means: work out this size)")
try:
    t.reshape(5, -1)
except RuntimeError as error:
    print("reshape(5, -1) RuntimeError:", error)

# Adding and removing size-1 dimensions.
seq = torch.tensor([5, 9, 2])
print("\nunsqueeze(0)  ", tuple(seq.unsqueeze(0).shape), "(a batch containing one sequence)")
print("unsqueeze(-1) ", tuple(seq.unsqueeze(-1).shape), "(each token gets its own row)")
print("squeeze()     ", tuple(seq.unsqueeze(0).squeeze().shape))

# Transposing swaps two axes. Multi-head attention (Chapter 14) splits the
# feature axis into (heads, features per head) and then moves the heads axis.
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)       # (batch, sequence, features)
split = x.reshape(2, 3, 2, 2)                      # (batch, sequence, heads, per_head)
moved = split.transpose(1, 2)                      # (batch, heads, sequence, per_head)
print("\nsplit features ", tuple(split.shape), "-> transpose(1, 2) ->", tuple(moved.shape))

# view() never copies, so it needs the elements laid out in order in memory.
# After a transpose they are not, so view() fails; reshape() copies if needed.
print("is_contiguous after transpose:", moved.is_contiguous())
try:
    moved.view(2, 2, 6)
except RuntimeError as error:
    print("view() failed:", str(error).split(".")[0])
print("reshape() works:", tuple(moved.reshape(2, 2, 6).shape),
      "| contiguous().view() works:", tuple(moved.contiguous().view(2, 2, 6).shape))

# reshape() is not transpose(): it keeps element order, transpose() changes it.
m = torch.arange(6).reshape(2, 3)
print("\nm            ", m.tolist())
print("m.reshape(3,2)", m.reshape(3, 2).tolist())
print("m.transpose   ", m.transpose(0, 1).tolist())
