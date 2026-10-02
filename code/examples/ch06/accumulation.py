"""Chapter 6.8: gradients add up. Accidental versus intentional accumulation.

Run from `code/`:  python examples/ch06/accumulation.py
"""

import torch
from torch import nn

from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data

loss_fn = nn.CrossEntropyLoss()
inputs, targets = make_band_data(64, seed=0)

# 1. backward() ADDS to .grad; it does not replace it.
torch.manual_seed(0)
model = TinyMLP(1, 8, 2)
loss_fn(model(inputs), targets).backward()
first = model.output.bias.grad.clone()
loss_fn(model(inputs), targets).backward()      # same batch again, no zero_grad
print("grad after one backward :", [round(v, 4) for v in first.tolist()])
print("grad after two backwards:", [round(v, 4) for v in model.output.bias.grad.tolist()], "(doubled)")

# 2. Intentional accumulation: 4 micro-batches of 16 give the same gradient as one batch of 64,
#    IF each micro-batch's loss is divided by the number of micro-batches.
torch.manual_seed(0)
full = TinyMLP(1, 8, 2)
torch.manual_seed(0)
accumulated = TinyMLP(1, 8, 2)

loss_fn(full(inputs), targets).backward()
micro_batches = 4
for chunk_inputs, chunk_targets in zip(inputs.chunk(micro_batches), targets.chunk(micro_batches)):
    (loss_fn(accumulated(chunk_inputs), chunk_targets) / micro_batches).backward()

same = all(torch.allclose(a.grad, b.grad, atol=1e-6) for a, b in zip(full.parameters(), accumulated.parameters()))
print("\n4 micro-batches (loss divided by 4) match one full batch:", same)

# Without dividing, the accumulated gradient is 4 times too large.
torch.manual_seed(0)
undivided = TinyMLP(1, 8, 2)
for chunk_inputs, chunk_targets in zip(inputs.chunk(micro_batches), targets.chunk(micro_batches)):
    loss_fn(undivided(chunk_inputs), chunk_targets).backward()
ratio = (undivided.output.bias.grad / full.output.bias.grad).tolist()
print("without dividing, gradient / full-batch gradient:", [round(v, 3) for v in ratio])
