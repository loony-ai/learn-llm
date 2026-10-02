"""Chapter 6.2: what the cross-entropy loss reports, by observation.

Run from `code/`:  python examples/ch06/cross_entropy.py
"""

import torch
from torch import nn

loss_fn = nn.CrossEntropyLoss()          # takes LOGITS and the index of the correct candidate
candidates = ["lit", "wrote", "rang", "slept"]
correct = torch.tensor([0])              # the right answer is candidate 0, "lit"

cases = {
    "confident and right": [6.0, 0.0, 0.0, 0.0],
    "leaning right":       [1.5, 0.5, 0.0, 0.0],
    "no preference":       [0.0, 0.0, 0.0, 0.0],
    "leaning wrong":       [0.0, 1.5, 0.5, 0.0],
    "confident and wrong": [0.0, 6.0, 0.0, 0.0],
}
print(f"{'case':<22}{'share for lit':>14}{'loss':>8}")
for label, logits in cases.items():
    logits = torch.tensor([logits])
    share = torch.softmax(logits, dim=-1)[0, 0].item()
    print(f"{label:<22}{share:>14.3f}{loss_fn(logits, correct).item():>8.3f}")

# A batch: the loss is the average over the examples.
batch = torch.tensor([[6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0]])
targets = torch.tensor([0, 0])
per_example = nn.CrossEntropyLoss(reduction="none")(batch, targets)
print("\nper-example losses:", [round(v, 3) for v in per_example.tolist()],
      "-> batch loss (their average):", round(loss_fn(batch, targets).item(), 3))

# A reference value: spreading evenly over N candidates gives the same loss every time.
for n in (2, 4, 82, 50_000):
    print(f"even spread over {n:>6} candidates -> loss {loss_fn(torch.zeros(1, n), torch.tensor([0])).item():.3f}")

# Common mistake: passing softmax shares instead of logits.
shares = torch.softmax(batch, dim=-1)
print("\nloss from logits (correct):", round(loss_fn(batch, targets).item(), 3),
      "| from softmax shares (wrong):", round(loss_fn(shares, targets).item(), 3))
