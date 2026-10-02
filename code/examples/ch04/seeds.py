"""Chapter 4.6: what a random seed does, and what it does not do.

Run from `code/`:  python examples/ch04/seeds.py
"""

import random

import numpy as np
import torch

from llmfp.experiment import set_seed

set_seed(42)
first = (random.random(), np.random.rand(), torch.rand(1).item())
set_seed(42)
second = (random.random(), np.random.rand(), torch.rand(1).item())
print("same seed, same draws        :", first == second)
print("  python, numpy, torch       :", [round(v, 6) for v in first])

set_seed(43)
third = (random.random(), np.random.rand(), torch.rand(1).item())
print("different seed, same draws   :", first == third)

# A seed fixes a SEQUENCE. Any extra draw shifts everything after it.
set_seed(42)
_ = torch.rand(1)                      # e.g. a new layer was added and initialized first
shifted = torch.rand(1).item()
print("extra draw before, same value:", shifted == first[2])

# Each library has its own generator. Seeding one does not seed the others.
torch.manual_seed(42)
random.seed(0)
print("torch seeded alone, torch    :", round(torch.rand(1).item(), 6), "(matches the first torch draw above)")
