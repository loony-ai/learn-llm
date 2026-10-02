"""Chapter 3.9: batching: adding a batch dimension, stack versus cat.

Run from `code/`:  python examples/ch03/batching.py
"""

import torch

a = torch.tensor([5, 9, 2, 5])
b = torch.tensor([5, 1, 4, 0])
c = torch.tensor([8, 9, 2, 6])

batch = torch.stack([a, b, c])               # new axis 0: (3, 4)
print("stack  ->", tuple(batch.shape))
joined = torch.cat([a, b, c])                # same axis, longer: (12,)
print("cat    ->", tuple(joined.shape))

single = a.unsqueeze(0)                      # a batch of one: (1, 4)
print("one example as a batch ->", tuple(single.shape))

try:
    torch.stack([a, torch.tensor([1, 2])])
except RuntimeError as error:
    print("stack of different lengths -> RuntimeError:", error)

# Models process a whole batch with one call. The same operation runs on every
# example independently, and results come back in the same order.
weights = torch.tensor([1, 10, 100, 1000])
print("batch * weights summed per row:", (batch * weights).sum(dim=1).tolist())
print("same as one at a time         :", [(x * weights).sum().item() for x in (a, b, c)])
