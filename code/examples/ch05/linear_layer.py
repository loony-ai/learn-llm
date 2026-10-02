"""Chapter 5.3: a linear layer is many units applied at once.

Run from `code/`:  python examples/ch05/linear_layer.py
"""

import torch
from torch import nn

torch.manual_seed(0)
layer = nn.Linear(in_features=3, out_features=2)   # 2 units, each reading 3 inputs
print("weight shape:", tuple(layer.weight.shape), "-> one row of 3 weights per unit")
print("bias shape  :", tuple(layer.bias.shape), "-> one bias per unit")

x = torch.tensor([[1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0]])                 # a batch of 2 examples, 3 features each
out = layer(x)
print("input shape :", tuple(x.shape), "-> output shape:", tuple(out.shape))

# The same result, one unit and one example at a time, with plain loops:
by_hand = torch.zeros(2, 2)
for example in range(2):
    for unit in range(2):
        total = layer.bias[unit]
        for feature in range(3):
            total = total + x[example, feature] * layer.weight[unit, feature]
        by_hand[example, unit] = total
def rounded(t):
    return [[round(v, 4) for v in row] for row in t.tolist()]


print("layer output:", rounded(out.detach()))
print("by hand     :", rounded(by_hand.detach()))
print("identical   :", torch.allclose(out, by_hand))

# The layer computes the same thing with one matrix multiplication, written @.
# For every example and every unit it multiplies matching positions and adds up.
manual = x @ layer.weight.T + layer.bias
print("x @ W.T + b :", torch.allclose(out, manual))

# A linear layer also works on more dimensions: it applies to the LAST axis only.
tokens = torch.randn(4, 7, 3)                       # (batch, sequence, features)
print("3-D input   :", tuple(tokens.shape), "->", tuple(layer(tokens).shape))
