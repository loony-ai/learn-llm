"""Chapter 6.5: automatic differentiation: what PyTorch records and computes.

Run from `code/`:  python examples/ch06/autograd_basics.py
"""

import torch

from llmfp.nn_basics import TinyMLP

a = torch.tensor(3.0, requires_grad=True)     # a value we may want to adjust
b = torch.tensor(4.0)                          # a fixed value
c = a * b
d = c + 2.0
print("requires_grad:", a.requires_grad, b.requires_grad, c.requires_grad, d.requires_grad)
print("recorded steps:", d.grad_fn.name(), "<-", d.grad_fn.next_functions[0][0].name())

d.backward()                                   # work backward through the recorded steps
print("a.grad:", a.grad.item(), "(raising a by a little raises d by about 4 times as much)")
print("b.grad:", b.grad, "(b was not marked, so no gradient)")

# For a network: every parameter gets a gradient tensor of exactly its own shape.
torch.manual_seed(0)
model = TinyMLP(1, 4, 2)
inputs, targets = torch.tensor([[2.0], [0.5]]), torch.tensor([1, 0])
loss = torch.nn.functional.cross_entropy(model(inputs), targets)
print("\nbefore backward, hidden.weight.grad is", model.hidden.weight.grad)
loss.backward()
for name, parameter in model.named_parameters():
    print(f"  {name:<14} shape={tuple(parameter.shape)!s:<8} grad shape={tuple(parameter.grad.shape)}")

# detach(): a tensor with the same values, cut off from the record.
print("\nlogits.detach().requires_grad:", model(inputs).detach().requires_grad)

# backward() frees the record after use; calling it twice is an error.
loss2 = torch.nn.functional.cross_entropy(model(inputs), targets)
loss2.backward()
try:
    loss2.backward()
except RuntimeError as error:
    print("second backward:", str(error).split(".")[0])
