"""Chapter 5.7: from scores (logits) to a ranked choice, by observing softmax.

Run from `code/`:  python examples/ch05/softmax_behavior.py
"""

import torch

candidates = ["lit", "wrote", "rang", "slept"]


def show(label: str, logits: torch.Tensor) -> None:
    shares = torch.softmax(logits, dim=-1)
    pairs = ", ".join(f"{w}={s:.3f}" for w, s in zip(candidates, shares.tolist()))
    print(f"{label:<28} logits={[round(v, 1) for v in logits.tolist()]!s:<26} -> {pairs}  (total {shares.sum():.3f})")


base = torch.tensor([2.0, 1.0, 0.0, -1.0])
show("original", base)
show("add 10 to every score", base + 10)
show("subtract 5 from every score", base - 5)
show("gaps doubled", base * 2)
show("gaps halved", base / 2)
show("all equal", torch.zeros(4))
show("one score far above", torch.tensor([10.0, 1.0, 0.0, -1.0]))

print("\nargmax is the same before and after softmax:", base.argmax().item(), torch.softmax(base, -1).argmax().item())

# Sampling: choose at random in proportion to the shares (Chapter 1's tickets in a hat).
torch.manual_seed(0)
draws = torch.multinomial(torch.softmax(base, -1), num_samples=1000, replacement=True)
counts = torch.bincount(draws, minlength=4)
print("1000 samples:", dict(zip(candidates, counts.tolist())))

# softmax works on a whole batch at once, along the axis you name.
batch = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 5.0, 0.0]])
print("batch rows each total 1:", [round(v, 4) for v in torch.softmax(batch, dim=-1).sum(dim=-1).tolist()])
print("wrong axis (dim=0) columns total 1 instead:", [round(v, 4) for v in torch.softmax(batch, dim=0).sum(dim=0).tolist()])
