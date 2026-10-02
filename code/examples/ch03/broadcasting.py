"""Chapter 3.7: broadcasting, and the silent bug it can cause.

Run from `code/`:  python examples/ch03/broadcasting.py
"""

import torch

scores = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])               # shape (2, 3)
print("scores + 10           ->", (scores + 10).tolist(), "(a single number is used everywhere)")

per_column = torch.tensor([100.0, 200.0, 300.0])       # shape (3,)
print("scores + per_column   ->", (scores + per_column).tolist(), "(added to every row)")

per_row = torch.tensor([[1000.0], [2000.0]])           # shape (2, 1)
print("scores + per_row      ->", (scores + per_row).tolist(), "(added to every column)")

try:
    scores + torch.tensor([1.0, 2.0])                  # shape (2,) does not line up with (2, 3)
except RuntimeError as error:
    print("scores + shape (2,)   -> RuntimeError:", error)

# THE SILENT BUG: a (3,) tensor and a (3, 1) tensor broadcast to (3, 3).
predictions = torch.tensor([1.0, 2.0, 3.0])            # shape (3,)
targets = torch.tensor([[1.0], [2.0], [3.0]])          # shape (3, 1), e.g. from a careless unsqueeze
difference = predictions - targets
print("\npredictions - targets shape:", tuple(difference.shape), "(expected (3,))")
print(difference)
print("mean absolute difference:", difference.abs().mean().item(), "(should be 0.0; no error was raised)")
print("after fixing the shape  :", (predictions - targets.squeeze(-1)).abs().mean().item())
