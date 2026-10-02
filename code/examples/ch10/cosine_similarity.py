"""Chapter 10.4: what cosine similarity reports, by observation.

Run from `code/`:  python examples/ch10/cosine_similarity.py
"""

import torch
from torch.nn import functional as F

a = torch.tensor([1.0, 2.0, 0.5])
cases = {
    "itself": a,
    "same direction, 10x longer": a * 10,
    "slightly different": torch.tensor([1.1, 1.9, 0.6]),
    "unrelated (at right angles)": torch.tensor([2.0, -1.0, 0.0]),
    "opposite direction": -a,
}
for label, b in cases.items():
    print(f"{label:<28} cosine similarity {F.cosine_similarity(a, b, dim=0).item():6.3f}")
