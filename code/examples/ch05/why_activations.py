"""Chapter 5.4: why stacked layers need an activation function between them.

Run from `code/`:  python examples/ch05/why_activations.py
"""

import torch
from torch import nn

from llmfp.nn_basics import TinyMLP

torch.manual_seed(0)
inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])   # evenly spaced inputs

# 1. Two linear layers with NOTHING between them.
plain = TinyMLP(1, 8, 1, activation="none")
out = plain(inputs).detach().squeeze(-1)
steps = out[1:] - out[:-1]
print("no activation : outputs", [round(v, 3) for v in out.tolist()])
print("                step between neighbors", [round(v, 3) for v in steps.tolist()], "(always the same)")

# The two layers can be merged into ONE equivalent linear layer.
merged = nn.Linear(1, 1)
with torch.no_grad():
    merged.weight.copy_(plain.output.weight @ plain.hidden.weight)
    merged.bias.copy_(plain.output.weight @ plain.hidden.bias + plain.output.bias)
print("                merged single layer gives the same outputs:", torch.allclose(merged(inputs), plain(inputs), atol=1e-6))

# 2. The same shape of network WITH ReLU between the layers.
bent = TinyMLP(1, 8, 1, activation="relu")
out = bent(inputs).detach().squeeze(-1)
steps = out[1:] - out[:-1]
print("\nwith ReLU     : outputs", [round(v, 3) for v in out.tolist()])
print("                step between neighbors", [round(v, 3) for v in steps.tolist()], "(can change)")

# 3. A hand-built ReLU network that responds only to inputs in a band (around 2).
#    No single linear layer can do this: its output can only rise or fall steadily.
band = TinyMLP(1, 3, 1, activation="relu")
with torch.no_grad():
    band.hidden.weight.copy_(torch.tensor([[1.0], [1.0], [1.0]]))
    band.hidden.bias.copy_(torch.tensor([-1.0, -2.0, -3.0]))
    band.output.weight.copy_(torch.tensor([[1.0, -2.0, 1.0]]))
    band.output.bias.zero_()
probe = torch.tensor([[0.0], [1.0], [1.5], [2.0], [2.5], [3.0], [4.0]])
print("\nband detector :")
for x, y in zip(probe.squeeze(-1).tolist(), band(probe).detach().squeeze(-1).tolist()):
    print(f"   input {x:3.1f} -> {y:4.2f} {'#' * round(y * 10)}")
