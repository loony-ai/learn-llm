"""Chapter 6.6: optimizers move parameters against their gradients. SGD versus AdamW.

Same one-knob task as nudge_one_knob.py. The best knob value is about 12.

Run from `code/`:  python examples/ch06/optimizer_steps.py
"""

import torch

crates = torch.tensor([1.0, 2.0, 3.0, 5.0, 8.0])
kilograms = torch.tensor([12.1, 23.8, 36.2, 59.9, 96.3])
loss_fn = torch.nn.MSELoss()


def run(make_optimizer, steps=8, start=5.0):
    knob = torch.tensor(start, requires_grad=True)
    optimizer = make_optimizer([knob])
    path = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(knob * crates, kilograms)
        loss.backward()
        optimizer.step()
        path.append(round(knob.item(), 3))
    return path


# The update rule written out by hand: move against the gradient, scaled by the learning rate.
knob = torch.tensor(5.0, requires_grad=True)
manual = []
for _ in range(8):
    knob.grad = None
    loss_fn(knob * crates, kilograms).backward()
    with torch.no_grad():                       # the update itself must not be recorded
        knob -= 0.005 * knob.grad
    manual.append(round(knob.item(), 3))
print("by hand, lr=0.005       :", manual)
print("SGD, lr=0.005           :", run(lambda p: torch.optim.SGD(p, lr=0.005)))
print("SGD, lr=0.0005 (small)  :", run(lambda p: torch.optim.SGD(p, lr=0.0005)))
print("SGD, lr=0.03 (large)    :", run(lambda p: torch.optim.SGD(p, lr=0.03)))
print("SGD, lr=0.06 (too large):", run(lambda p: torch.optim.SGD(p, lr=0.06)))
adamw = run(lambda p: torch.optim.AdamW(p, lr=1.0, weight_decay=0.0), steps=40)
print("AdamW, lr=1.0, every 4th of 40 steps:", adamw[::4])

# Keep the diverging run going: values grow until they overflow to infinity, then become NaN.
knob = torch.tensor(5.0, requires_grad=True)
optimizer = torch.optim.SGD([knob], lr=0.06)
seen_infinite = False
for step in range(1, 1001):
    optimizer.zero_grad()
    loss = loss_fn(knob * crates, kilograms)
    loss.backward()
    optimizer.step()
    if torch.isinf(loss) and not seen_infinite:
        print(f"lr=0.06: step {step}: loss={loss.item()}  knob={knob.item():.3g}")
        seen_infinite = True
    if torch.isnan(loss) or torch.isnan(knob):
        print(f"lr=0.06: step {step}: loss={loss.item()}  knob={knob.item()}")
        break
