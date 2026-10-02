"""Chapter 6.9: training mode, evaluation mode, and switching off gradient tracking.

Run from `code/`:  python examples/ch06/modes.py
"""

import torch
from torch import nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(8, 2))
x = torch.ones(1, 4)

model.train()
print("train mode, same input three times:", [[round(v, 3) for v in model(x)[0].tolist()] for _ in range(3)])
model.eval()
print("eval mode, same input three times :", [[round(v, 3) for v in model(x)[0].tolist()] for _ in range(3)])

# Dropout in training mode zeroes random values (here half) and scales up the rest.
dropout = nn.Dropout(p=0.5)
dropout.train()
print("\ndropout, training mode:", dropout(torch.ones(10)).tolist())
dropout.eval()
print("dropout, eval mode    :", dropout(torch.ones(10)).tolist())

# Gradient tracking: on by default for computations involving parameters.
model.eval()
tracked = model(x)
with torch.no_grad():
    untracked = model(x)
with torch.inference_mode():
    inference = model(x)
print("\nrecorded for backward? normal:", tracked.requires_grad,
      "| no_grad:", untracked.requires_grad, "| inference_mode:", inference.requires_grad)

# eval() and no_grad() are independent: one changes layer behavior, the other bookkeeping.
model.train()
with torch.no_grad():
    outputs = [round(model(x)[0, 0].item(), 3) for _ in range(3)]
print("no_grad but still train mode (dropout active):", outputs)
