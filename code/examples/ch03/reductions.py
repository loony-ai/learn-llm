"""Chapter 3.8: operations along an axis: sum, mean, max, argmax.

Run from `code/`:  python examples/ch03/reductions.py
"""

import torch

# Scores for 5 candidate tokens, for 2 sequences.
scores = torch.tensor([[0.5, 2.5, 0.25, 1.0, -1.0],
                       [1.5, 0.25, 3.0, 0.0, 0.5]])
print("scores shape          :", tuple(scores.shape))
print("sum over everything   :", round(scores.sum().item(), 4))
print("mean along dim=1      :", [round(v, 3) for v in scores.mean(dim=1).tolist()], "-> one value per sequence (rounded)")
print("max along dim=1       :", scores.max(dim=1).values.tolist())
print("argmax along dim=1    :", scores.argmax(dim=1).tolist(), "-> position of the top score = greedy choice")
print("keepdim=True shape    :", tuple(scores.max(dim=1, keepdim=True).values.shape), "(axis kept with size 1)")

# keepdim matters for broadcasting: subtract each row's maximum from that row.
row_max = scores.max(dim=1, keepdim=True).values
print("scores - row max      :", (scores - row_max).tolist())
top = scores.topk(2, dim=1)
print("topk(2) values/indices:", top.values.tolist(), top.indices.tolist())
