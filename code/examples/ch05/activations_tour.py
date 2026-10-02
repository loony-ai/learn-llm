"""Chapter 5.4: what common activation functions do to a range of inputs.

Run from `code/`:  python examples/ch05/activations_tour.py
"""

import torch

from llmfp.nn_basics import ACTIVATIONS

x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
print(f"{'input':>8}" + "".join(f"{v:>8.1f}" for v in x.tolist()))
for name in ("relu", "gelu", "tanh", "sigmoid"):
    y = ACTIVATIONS[name]()(x)
    print(f"{name:>8}" + "".join(f"{v:>8.3f}" for v in y.tolist()))
