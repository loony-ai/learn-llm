"""Chapter 3.5: indexing and slicing.

Run from `code/`:  python examples/ch03/indexing.py
"""

import torch

batch = torch.tensor([[10, 11, 12, 13, 14],
                      [20, 21, 22, 23, 24],
                      [30, 31, 32, 33, 34]])
print("batch[1]          ->", batch[1].tolist(), "(one row)")
print("batch[1, 3]       ->", batch[1, 3].item(), "(one element)")
print("batch[:, 0]       ->", batch[:, 0].tolist(), "(first column: position 0 of every sequence)")
print("batch[:, -1]      ->", batch[:, -1].tolist(), "(last position of every sequence)")
print("batch[:, :-1]     ->", batch[:, :-1].tolist(), "(all but the last position)")
print("batch[:, 1:]      ->", batch[:, 1:].tolist(), "(all but the first position)")
print("batch[0, ::2]     ->", batch[0, ::2].tolist(), "(every second element)")

# Indexing with a list or tensor of positions picks several rows at once.
# This is exactly how an embedding table is looked up (Chapter 10).
table = torch.tensor([[0.0, 0.0], [0.25, 0.5], [0.75, 1.0], [1.25, 1.5]])  # 4 rows of 2 numbers
ids = torch.tensor([3, 1, 3])
print("\ntable[ids]        ->", table[ids].tolist(), "shape", tuple(table[ids].shape))

# A boolean mask selects the elements where it is True.
mask = batch > 22
print("batch > 22        ->", mask[1].tolist(), "(row 1 of the mask)")
print("batch[mask]       ->", batch[mask].tolist())

# A slice is a VIEW: it shares memory with the original. Changing it changes both.
row = batch[0]
row[0] = 99
print("\nafter row[0] = 99, batch[0] ->", batch[0].tolist(), "(the original changed)")
copy = batch[1].clone()
copy[0] = -1
print("after copy[0] = -1, batch[1] ->", batch[1].tolist(), "(clone() made an independent copy)")

try:
    batch[3]
except IndexError as error:
    print("\nbatch[3] -> IndexError:", error)
