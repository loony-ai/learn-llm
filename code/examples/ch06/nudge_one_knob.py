"""Chapter 6.3-6.4: nudge one adjustable number and watch the loss; compare with the gradient.

Task: the harbor weighs fish by the crate. Learn "kilograms per crate" from records.
The model has ONE parameter, the knob. Prediction = knob times the number of crates.

Run from `code/`:  python examples/ch06/nudge_one_knob.py
"""

import torch

crates = torch.tensor([1.0, 2.0, 3.0, 5.0, 8.0])
kilograms = torch.tensor([12.1, 23.8, 36.2, 59.9, 96.3])     # about 12 kg per crate
loss_fn = torch.nn.MSELoss()   # average squared miss: big misses count much more than small ones


def loss_at(knob_value: float) -> float:
    return loss_fn(knob_value * crates, kilograms).item()


print("Trying knob values by hand:")
for value in (5.0, 10.0, 11.0, 12.0, 13.0, 15.0):
    print(f"  knob={value:5.1f}  loss={loss_at(value):9.3f}")

# Nudge-and-observe at knob = 10: does a small increase raise or lower the loss?
knob, nudge = 10.0, 0.001
change = loss_at(knob + nudge) - loss_at(knob)
print(f"\nAt knob=10, raising it by {nudge} changes the loss by {change:.4f}")
print(f"  -> loss change per unit of knob change: {change / nudge:.1f}  (negative: increasing the knob helps)")

# Automatic differentiation reports the same thing for every parameter at once.
knob_tensor = torch.tensor(10.0, requires_grad=True)
loss = loss_fn(knob_tensor * crates, kilograms)
loss.backward()
print(f"  -> PyTorch's gradient at knob=10: {knob_tensor.grad.item():.1f}")

for value in (12.0, 14.0):
    t = torch.tensor(value, requires_grad=True)
    loss_fn(t * crates, kilograms).backward()
    print(f"     gradient at knob={value}: {t.grad.item():7.1f}")
