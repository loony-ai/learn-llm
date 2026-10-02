"""Chapter 5.5: nn.Module, parameters, and the forward pass.

Run from `code/`:  python examples/ch05/module_basics.py
"""

import torch
from torch import nn


class UrgencyScorer(nn.Module):
    """Scores messages as (not urgent, urgent) from 3 features."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(3, 4)
        self.output = nn.Linear(4, 2)
        self.scale = 2.0                       # a plain attribute: NOT a parameter

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.hidden(features))
        return self.output(hidden) * self.scale


def rounded(t):
    return [[round(v, 3) for v in row] for row in t.detach().tolist()]


torch.manual_seed(0)
model = UrgencyScorer()
print(model)
print("\nnamed parameters:")
for name, parameter in model.named_parameters():
    print(f"  {name:<15} shape={tuple(parameter.shape)}  requires_grad={parameter.requires_grad}")

batch = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
scores = model(batch)                        # calls __call__, which runs forward
print("\nscores:", rounded(scores), "shape", tuple(scores.shape))

# The same architecture with a different seed has different parameters,
# so it produces different scores for the same input.
torch.manual_seed(1)
other = UrgencyScorer()
print("other initialization:", rounded(other(batch)))

# nn.Sequential builds simple chains without writing a class.
chain = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
print("\nSequential:", chain, "-> output shape", tuple(chain(batch).shape))

# Forgetting super().__init__() breaks parameter registration immediately.
class Broken(nn.Module):
    def __init__(self) -> None:
        self.layer = nn.Linear(3, 2)

try:
    Broken()
except AttributeError as error:
    print("\nwithout super().__init__():", error)
