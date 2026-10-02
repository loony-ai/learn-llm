## Chapter 6: How Training Works: Loss, Gradients, and Optimizers

[Back to index](../../README.md) · Previous: [Chapter 5](ch05-neural-networks.md) · Next: Chapter 7 (planned)

In Chapter 5 you set a network's parameters by hand to detect a band of input values. It took careful thought for three hidden units. A language model has millions to billions of parameters, and nobody can set them by hand. **Training** is the automated process that sets them from examples. It is often presented with calculus. This chapter presents it as what the software actually does, step by step, with every step observed in code:

1. Run the network on a batch of examples (the forward pass).
2. Measure, with one number, how wrong its outputs are (the loss).
3. For every parameter, work out which direction to move it, and how strongly the loss responds to it (the gradient, computed by the backward pass).
4. Move every parameter a small step in the direction that reduces the loss (the optimizer step).
5. Repeat, many thousands of times.

By the end of the chapter, a network will have found the band from Chapter 5 by itself, from labeled examples. You will also have seen, deliberately, the most common ways training goes wrong.

> **What this chapter does and does not give you.** You will understand training operationally: what is measured, what is changed, and what behavior results. The mathematics of *why* gradients point the right way, and of how optimizers are derived, is left out on purpose. That theory matters for research on new training methods. It is not needed to implement, run, debug, and adapt standard training, which is what this book teaches.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain what a loss measures, and read cross-entropy loss values by comparing them with reference points.
2. Explain what a gradient tells you about a parameter, and confirm it by nudging the parameter and watching the loss.
3. Describe what automatic differentiation records and computes, and use `backward()` and `.grad`.
4. Describe what an optimizer and a learning rate do, and distinguish SGD from AdamW by their observed behavior.
5. Write a complete training loop with minibatches, epochs, and validation, and explain each line.
6. Explain why gradients accumulate, why `zero_grad` is needed, and how to accumulate gradients deliberately and correctly.
7. Use training mode, evaluation mode, and `no_grad` correctly, and say what each one controls.
8. Recognize learning rates that are too small, too large, and workable from their effect on training.

#### Prerequisites

- [Chapter 4](ch04-data-experiments-reproducibility.md): training and validation splits, overfitting, baselines, seeds, run records.
- [Chapter 5](ch05-neural-networks.md): units, linear layers, activations, `nn.Module`, logits, softmax, `TinyMLP`, and the band detector (section 5.4).

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Loss | One number measuring how wrong a model's outputs are on some examples; lower is better | 6.2 |
| Cross-entropy loss | The standard loss for choosing among candidates: large when the correct candidate gets a small share | 6.2 |
| Mean squared error | A loss for predicting numbers: the average of squared misses, so big misses count much more | 6.3 |
| Gradient | For each parameter: which direction would increase the loss, and how strongly the loss responds | 6.4 |
| Automatic differentiation (autograd) | PyTorch's system for recording computations and computing gradients from that record | 6.5 |
| Backward pass, backpropagation | Working backward through the recorded computation to compute every gradient | 6.5 |
| Optimizer | The component that changes parameters using their gradients | 6.6 |
| Learning rate | The setting that controls how large the optimizer's steps are | 6.6 |
| SGD, AdamW | Two optimizers: a plain step against the gradient / an adaptive step with momentum and weight decay | 6.6 |
| Minibatch, epoch | A subset of training examples used for one step / one full pass over the training data | 6.7 |
| Gradient accumulation | Adding up gradients from several minibatches before one update, on purpose | 6.8 |
| Training mode / evaluation mode | Module settings that switch training-only behavior (such as dropout) on or off | 6.9 |
| Dropout | Randomly zeroing some values during training, to make the network less dependent on any one of them | 6.9 |
| NaN, infinity | Special float values meaning "not a number" and "too large to represent"; they signal broken training | 6.6, 6.10 |

---

### 6.1 The problem: how does software adjust millions of numbers sensibly?

Imagine the simplest possible strategy: change one parameter at random, keep the change if the model got better, undo it if not. For a network with a few thousand parameters, each "try" needs a full evaluation, and almost every random change makes things slightly worse. For a billion parameters, this would never finish.

What would make adjustment efficient?

- A **single number to improve**, so that "better" and "worse" are unambiguous. That is the **loss** (6.2).
- For **every parameter at once**, a report of which direction to move it and how much it matters. That is the **gradient** (6.3–6.4).
- A way to compute all of those reports **at about the cost of one extra pass through the network**, rather than one evaluation per parameter. That is **automatic differentiation** (6.5).
- A rule for **how far to move** each parameter given its report. That is the **optimizer** and its **learning rate** (6.6).

The rest of the chapter takes these one at a time, then assembles them into a training loop (6.7).

---

### 6.2 Loss: one number that measures how wrong the predictions are

A **loss** turns a model's outputs and the correct answers into one number. Lower means better. Training is the search for parameter values that make the loss on the training data low.

For tasks where the model chooses among candidates, which includes every language model in this book, the standard loss is **cross-entropy**. Rather than define it with a formula, here is what it reports:

File: [`code/examples/ch06/cross_entropy.py`](../../code/examples/ch06/cross_entropy.py)

```python
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
```

Observed output:

```text
case                   share for lit    loss
confident and right            0.993   0.007
leaning right                  0.551   0.596
no preference                  0.250   1.386
leaning wrong                  0.123   2.096
confident and wrong            0.002   6.007

per-example losses: [0.007, 6.007] -> batch loss (their average): 3.007
even spread over      2 candidates -> loss 0.693
even spread over      4 candidates -> loss 1.386
even spread over     82 candidates -> loss 4.407
even spread over  50000 candidates -> loss 10.820

loss from logits (correct): 3.007 | from softmax shares (wrong): 1.244
```

How to read cross-entropy:

- **It depends on the share the model gives to the correct answer.** The model's logits are turned into shares by softmax (Chapter 5.7); the loss looks at the share for the correct candidate. A share near 1 gives a loss near 0. As the share falls toward 0, the loss grows without limit: 0.596 when leaning right, 2.096 when leaning wrong, 6.007 when confidently wrong.
- **Confident mistakes are punished hardest.** Being confidently wrong costs far more than being unsure (1.386). This pushes training to avoid overconfidence on things the model gets wrong.
- **The loss of a batch is the average over its examples.** One confident right answer and one confident wrong answer average to about 3.
- **Even spread gives a fixed reference value for each number of candidates.** Spreading shares evenly over 2 candidates costs 0.693; over 4, 1.386; over the 82 harbor words, 4.407; over a 50,000-token vocabulary, 10.820. These are the losses of a model that knows nothing. A trained model's loss should be well below the reference for its vocabulary size, and Chapter 19.3 uses these reference points to interpret language-model training curves.
- **PyTorch's cross-entropy takes logits, not shares.** It applies softmax internally, in a numerically careful way. Passing it softmax outputs applies softmax twice and gives a wrong, misleadingly small loss (1.244 instead of 3.007), with no error. This is one of the most common silent bugs in training code.

The targets are given as the *index* of the correct candidate (an int64 tensor with one entry per example), not as a list of shares. For a language model, the target at each position is the token ID of the actual next token.

---

### 6.3 Nudge and observe, with a single adjustable number

To see what a gradient is, start with a model that has exactly one parameter. The harbor records how many crates of fish came in and how many kilograms they weighed. The model predicts kilograms as "the knob times the number of crates"; the knob is its only parameter. The loss here is **mean squared error**, the usual loss for predicting numbers rather than choosing among candidates: it squares each miss, which makes large misses count much more than small ones, and averages them.

File: [`code/examples/ch06/nudge_one_knob.py`](../../code/examples/ch06/nudge_one_knob.py)

```python
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
```

Observed output:

```text
Trying knob values by hand:
  knob=  5.0  loss= 1015.598
  knob= 10.0  loss=   84.198
  knob= 11.0  loss=   21.518
  knob= 12.0  loss=    0.038
  knob= 13.0  loss=   19.758
  knob= 15.0  loss=  182.798

At knob=10, raising it by 0.001 changes the loss by -0.0833
  -> loss change per unit of knob change: -83.3  (negative: increasing the knob helps)
  -> PyTorch's gradient at knob=10: -83.3
     gradient at knob=12.0:    -0.9
     gradient at knob=14.0:    81.5
```

The first table is training by trial: the loss is lowest near a knob of 12 (about 12 kg per crate), and grows on both sides.

The second part is the key idea. Standing at knob 10, we **nudge** it up by a tiny amount and observe that the loss goes **down**, by about 83 times the size of the nudge. That one measurement tells us two things:

- **Direction:** increasing the knob reduces the loss, so we should increase it.
- **Sensitivity:** the loss responds strongly to this knob here. A small change makes a big difference.

The third part asks PyTorch for the **gradient** of the loss with respect to the knob, and gets -83.3: exactly the measured response. At knob 12, near the best value, the gradient is close to zero (the loss barely changes when you nudge it). At knob 14, it is positive: increasing the knob would *increase* the loss, so we should decrease it.

---

### 6.4 Gradients: a direction-and-sensitivity report for every parameter

The **gradient** of the loss with respect to a parameter is precisely the report you just measured: if this parameter were increased by a tiny amount, how much would the loss change, per unit of change? Its **sign** gives the direction (positive: increasing the parameter increases the loss; negative: increasing it decreases the loss). Its **size** gives the sensitivity.

For a network, there is one such number for every individual parameter. So the gradient for a weight tensor of shape `(16, 1)` is another tensor of shape `(16, 1)`: one report per weight. Section 6.5 shows this.

Why not measure gradients by nudging, as in 6.3? Because nudging measures one parameter at a time. A network with a million parameters would need a million extra evaluations for one step of training. Automatic differentiation computes every gradient at once, for about the cost of one extra pass through the network. That difference is what makes training large networks possible at all.

Two cautions about gradients:

- **A gradient describes the immediate neighborhood only.** It says what a *tiny* change would do from the current values. A large step can overshoot, which is why step size matters (6.6).
- **A gradient is computed for the current batch.** Different batches give somewhat different gradients. Each is a noisy estimate of the gradient for the whole training set, which is fine as long as steps are small and many (6.7).

---

### 6.5 Automatic differentiation: PyTorch records operations and computes gradients for you

File: [`code/examples/ch06/autograd_basics.py`](../../code/examples/ch06/autograd_basics.py)

```python
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
```

Observed output:

```text
requires_grad: True False True True
recorded steps: AddBackward0 <- MulBackward0
a.grad: 4.0 (raising a by a little raises d by about 4 times as much)
b.grad: None (b was not marked, so no gradient)

before backward, hidden.weight.grad is None
  hidden.weight  shape=(4, 1)   grad shape=(4, 1)
  hidden.bias    shape=(4,)     grad shape=(4,)
  output.weight  shape=(2, 4)   grad shape=(2, 4)
  output.bias    shape=(2,)     grad shape=(2,)

logits.detach().requires_grad: False
second backward: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)
```

What happened, operationally:

1. **Marking.** `requires_grad=True` marks a tensor as something we may want gradients for. Every `nn.Module` parameter is marked automatically (you saw `requires_grad=True` in Chapter 5.5). Ordinary data, like `b` or a batch of inputs, is not.
2. **Recording.** When an operation involves a marked tensor, PyTorch records it: each result keeps a reference (`grad_fn`) to the step that produced it, and that step keeps references to its inputs. `d` was made by an addition (`AddBackward0`), whose input was made by a multiplication (`MulBackward0`). The record is a chain, or for a network a branching graph, from the parameters to the loss. This record is called the *computation graph*.
3. **Backward.** `loss.backward()` walks that record backward, from the loss to every marked tensor. Each recorded step knows how its output responds to small changes in its inputs (for a multiplication by 4, a change in the input produces 4 times that change in the output). Combining those step-by-step responses along the way back gives, for each marked tensor, how the *loss* responds to it. This backward walk is the **backward pass**, also called **backpropagation**. The result is stored in each tensor's `.grad`: `a.grad` is 4, and `b`, which was not marked, has none.
4. **Shapes.** Every parameter's gradient has exactly the parameter's shape.

Three practical details:

- **`.grad` is empty (`None`) until the first backward pass.**
- **`detach()`** returns a tensor with the same values but cut off from the record, so nothing computed from it will contribute gradients. Use it when you need values only, for logging or plotting.
- **The record is freed after `backward()`** to save memory, so calling `backward()` twice on the same loss is an error. Each training step builds a fresh record during its forward pass.

That the step-by-step combination gives exactly the right gradient is a mathematical fact (it is an application of calculus). You do not need it to use autograd correctly; you need to know what is recorded, when, and what ends up in `.grad`.

---

### 6.6 Optimizers and the learning rate

An **optimizer** changes each parameter using its gradient. The **learning rate** controls the size of the changes.

File: [`code/examples/ch06/optimizer_steps.py`](../../code/examples/ch06/optimizer_steps.py)

```python
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
```

Observed output:

```text
by hand, lr=0.005       : [6.446, 7.595, 8.507, 9.231, 9.806, 10.262, 10.624, 10.912]
SGD, lr=0.005           : [6.446, 7.595, 8.507, 9.231, 9.806, 10.262, 10.624, 10.912]
SGD, lr=0.0005 (small)  : [5.145, 5.286, 5.425, 5.561, 5.694, 5.824, 5.952, 6.077]
SGD, lr=0.03 (large)    : [13.678, 11.63, 12.114, 12.0, 12.026, 12.02, 12.022, 12.021]
SGD, lr=0.06 (too large): [22.357, -3.192, 34.416, -20.944, 60.546, -59.407, 117.163, -142.747]
AdamW, lr=1.0, every 4th of 40 steps: [6.0, 9.855, 12.938, 14.375, 14.11, 12.915, 11.657, 10.989, 11.098, 11.695]
lr=0.06: step 105: loss=inf  knob=3e+18
lr=0.06: step 217: loss=inf  knob=nan
```

**SGD by hand.** The first line updates the knob with plain code: subtract the learning rate times the gradient, inside `torch.no_grad()` so that the update itself is not recorded as part of the next computation. Moving *against* the gradient moves the parameter in the direction that lowers the loss. Starting from 5, the knob climbs toward 12. The second line uses PyTorch's `torch.optim.SGD` and produces identical numbers: that is all plain SGD does. **SGD** stands for *stochastic gradient descent*: "gradient descent" for stepping downhill along the gradient, and "stochastic" (random) because, in practice, each step uses the gradient of a randomly chosen minibatch (6.7).

**The learning rate decides everything about the path:**

- **Too small (0.0005):** every step is tiny; after 8 steps the knob has barely moved from 5. Training would eventually get there, but slowly and at great cost.
- **Workable (0.005):** steady progress toward 12.
- **Large (0.03):** the first step overshoots to 13.7, the next undershoots, and the knob then settles at 12. Fast, but on the edge.
- **Too large (0.06):** each step overshoots by more than the last. The knob swings ever more wildly: 22, -3, 34, -21, 61, and so on. This is **divergence**. Left running, the values become so large that the loss overflows to **infinity** (`inf`) at step 105, and the knob itself becomes **NaN** ("not a number", the float value produced by operations with no meaningful result, such as infinity minus infinity) at step 217. Once a NaN appears, it spreads through every computation it touches, and training is over. Chapter 20.5 covers how to detect and recover from this.

**AdamW behaves differently.** With a learning rate of 1.0, which would be absurd for SGD on this problem, AdamW moves the knob by about 1 per step at first, whatever the gradient's size. Two mechanisms explain the path you see:

- **Adaptive step size.** AdamW keeps, for each parameter, a running estimate of how large its gradients typically are, and scales the step accordingly, so steps are roughly the size of the learning rate whether the gradients are huge or tiny. That makes one learning rate work across parameters whose gradients differ enormously in size, which is the normal situation in deep networks.
- **Momentum.** It also keeps a running average of recent gradient directions, and steps in that averaged direction. A ball rolling downhill is a reasonable analogy, and also where it ends: momentum carries the knob past 12, to about 14.4, before the accumulated direction reverses and it comes back.

The "W" refers to *weight decay*: a small pull of every parameter toward zero on each step, which discourages unnecessarily large parameter values. The examples here turn it off (`weight_decay=0.0`) to keep the paths simple; Chapter 19.5 discusses it. AdamW is the default optimizer for training transformers, and this book uses it from here on unless stated otherwise.

---

### 6.7 The training loop, line by line

Now the pieces assemble into the loop used for every model in Part 1.

File: [`code/llmfp/training_basics.py`](../../code/llmfp/training_basics.py)

```python
"""A minimal, readable training loop for classification (Chapter 6).

    iterate_minibatches  yield (inputs, targets) batches, shuffled reproducibly
    train_step           one forward pass, loss, backward pass, and parameter update
    evaluate             loss and accuracy with training-only behavior switched off
    fit                  repeat train_step over the data for several epochs, with validation

Chapter 19 replaces this with a full pretraining loop (schedules, clipping,
checkpoints, mixed precision). The core steps stay exactly the same.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def iterate_minibatches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield matching slices of inputs and targets. The last batch may be smaller."""
    if len(inputs) != len(targets):
        raise ValueError(f"{len(inputs)} inputs but {len(targets)} targets")
    order = torch.randperm(len(inputs), generator=generator) if shuffle else torch.arange(len(inputs))
    for start in range(0, len(inputs), batch_size):
        chosen = order[start : start + batch_size]
        yield inputs[chosen], targets[chosen]


def train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One update. Returns the loss measured BEFORE the update."""
    model.train()                      # enable training-only behavior (e.g. dropout)
    optimizer.zero_grad()              # 1. clear gradients left over from the previous step
    logits = model(inputs)             # 2. forward pass: compute scores
    loss = loss_fn(logits, targets)    # 3. measure how wrong the scores are
    loss.backward()                    # 4. backward pass: compute a gradient for every parameter
    optimizer.step()                   # 5. adjust every parameter using its gradient
    return loss.item()


@torch.no_grad()  # decorator form: no gradient bookkeeping anywhere in this function
def evaluate(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, loss_fn: LossFunction, batch_size: int = 1024
) -> dict[str, float]:
    """Average loss and accuracy over a dataset, with the model in evaluation mode."""
    was_training = model.training
    model.eval()                       # disable training-only behavior
    total_loss, correct = 0.0, 0
    for batch_inputs, batch_targets in iterate_minibatches(inputs, targets, batch_size, shuffle=False):
        logits = model(batch_inputs)
        total_loss += loss_fn(logits, batch_targets).item() * len(batch_targets)
        correct += (logits.argmax(dim=-1) == batch_targets).sum().item()
    model.train(was_training)          # restore whatever mode the caller had
    return {"loss": total_loss / len(targets), "accuracy": correct / len(targets)}


def fit(
    model: nn.Module,
    train_data: tuple[torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor],
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Train for `epochs` passes over the training data. Returns one record per epoch.

    Stops early if the training loss becomes NaN or infinite, which means training
    has broken down (Chapter 6.10 shows how a too-large learning rate causes it).
    """
    generator = torch.Generator().manual_seed(seed)  # its own generator: shuffling is reproducible
    history = []
    for epoch in range(1, epochs + 1):
        losses = [
            train_step(model, x, y, loss_fn, optimizer)
            for x, y in iterate_minibatches(*train_data, batch_size, shuffle=True, generator=generator)
        ]
        train_loss = sum(losses) / len(losses)
        validation = evaluate(model, *validation_data, loss_fn)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": validation["loss"], "val_accuracy": validation["accuracy"]}
        history.append(record)
        logger.info("epoch %d: %s", epoch, record)
        if not math.isfinite(train_loss):
            logger.warning("training loss is %s at epoch %d; stopping", train_loss, epoch)
            break
    return history
```

#### `train_step`: the five lines at the heart of all training

1. **`model.train()`** puts the model in training mode (6.9).
2. **`optimizer.zero_grad()`** clears the gradients left from the previous step (6.8 explains why this is essential).
3. **`logits = model(inputs)`** is the forward pass, recording the computation.
4. **`loss = loss_fn(logits, targets)`** measures how wrong the outputs are.
5. **`loss.backward()`** computes a gradient for every parameter, and **`optimizer.step()`** moves every parameter using its gradient.

It returns `loss.item()`: the loss as a plain Python number, measured *before* this step's update. `.item()` also matters for memory: keeping the loss tensor itself would keep its whole computation record alive.

#### Minibatches and epochs

Computing the loss on all training examples for every step would be slow, and is unnecessary: the gradient from a random subset is a good enough estimate when steps are small. Each step therefore uses a **minibatch** (usually just called a *batch*), and `iterate_minibatches` deals them out in a shuffled order, with its own seeded generator so the order is reproducible. One full pass through the training data is an **epoch**. Shuffling every epoch means the batches differ each time, which is the "stochastic" in SGD.

#### `evaluate` and `fit`

`evaluate` computes loss and accuracy on a dataset with training-only behavior switched off and no gradient bookkeeping (6.9). `fit` runs epochs of training steps, evaluates on the validation set after each epoch, records the results, and stops if the training loss becomes NaN or infinite, which means training has broken down (6.10).

#### Milestone: training finds the band

The task: inputs between 0 and 4, labeled 1 inside the band 1.5–2.5 and 0 elsewhere. The data comes from [`code/llmfp/toy_data.py`](../../code/llmfp/toy_data.py):

```python
"""Small synthetic datasets for learning how training works (Chapter 6).

The band task: inputs are single numbers between 0 and 4; the label is 1 when the
input lies inside a band (default 1.5 to 2.5) and 0 otherwise. Chapter 5 built a
band detector by hand; Chapter 6 trains a network to find one from examples.
"""

from __future__ import annotations

import torch


def make_band_data(
    count: int, low: float = 1.5, high: float = 2.5, seed: int = 0, span: float = 4.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inputs of shape (count, 1), float32, and labels of shape (count,), int64."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.rand(count, 1, generator=generator) * span
    labels = ((inputs[:, 0] >= low) & (inputs[:, 0] <= high)).long()
    return inputs, labels
```

The script, driven by a configuration file and recorded as a run (Chapter 4):

File: [`code/configs/band-cpu.toml`](../../code/configs/band-cpu.toml)

```toml
# Chapter 6: train a small network to detect a band of input values.
runs_root = "runs"
run_name = "ch06-band"
seed = 0
train_examples = 2000
validation_examples = 500
hidden = 16
activation = "relu"
optimizer = "adamw"        # "adamw" or "sgd"
learning_rate = 0.01
epochs = 30
batch_size = 32
```

File: [`code/scripts/ch06_train_band.py`](../../code/scripts/ch06_train_band.py)

```python
"""Chapter 6 milestone: train a network to find the band that Chapter 5 built by hand.

Run from `code/`:
    python -m scripts.ch06_train_band
    python -m scripts.ch06_train_band --set optimizer=sgd --set learning_rate=0.5
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import torch
from torch import nn

from llmfp.config import ConfigError, load_config
from llmfp.experiment import finish_run, set_seed, start_run
from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data
from llmfp.training_basics import evaluate, fit


@dataclass(frozen=True)
class BandConfig:
    runs_root: str = "runs"
    run_name: str = "ch06-band"
    seed: int = 0
    train_examples: int = 2000
    validation_examples: int = 500
    hidden: int = 16
    activation: str = "relu"
    optimizer: str = "adamw"
    learning_rate: float = 0.01
    epochs: int = 30
    batch_size: int = 32


def make_optimizer(name: str, parameters, learning_rate: float) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/band-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--every", type=int, default=5, help="print every Nth epoch")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        config = load_config(BandConfig, args.config, args.overrides)
    except (ConfigError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}")

    set_seed(config.seed)
    run_dir = start_run(config.runs_root, config.run_name, config)
    train_data = make_band_data(config.train_examples, seed=config.seed)
    validation_data = make_band_data(config.validation_examples, seed=config.seed + 1)

    model = TinyMLP(1, config.hidden, 2, activation=config.activation)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config.optimizer, model.parameters(), config.learning_rate)

    # Two reference points before training: the untrained model, and the simplest
    # possible rule (always answer the most common label).
    majority = int(train_data[1].float().mean() < 0.5) ^ 1
    baseline_accuracy = (validation_data[1] == majority).float().mean().item()
    before = evaluate(model, *validation_data, loss_fn)
    print(f"Run: {run_dir}")
    print(f"Baseline 'always answer {majority}': val accuracy {baseline_accuracy:.1%}")
    print(f"Untrained model: val loss {before['loss']:.3f}, val accuracy {before['accuracy']:.1%}")

    history = fit(model, train_data, validation_data, loss_fn, optimizer, config.epochs, config.batch_size, config.seed)
    print(f"{'epoch':>5} | {'train loss':>10} | {'val loss':>8} | {'val acc':>7}")
    for record in history:
        if record["epoch"] % args.every == 0 or record["epoch"] in (1, len(history)):
            print(f"{record['epoch']:>5} | {record['train_loss']:>10.4f} | {record['val_loss']:>8.4f} | {record['val_accuracy']:>7.1%}")

    # What did it learn? Ask for its choice across the input range.
    probe = torch.arange(0.0, 4.01, 0.25).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        choices = model(probe).argmax(dim=-1).tolist()
    print("Learned answer by input (1 = inside band):")
    print("  " + " ".join(f"{x:.2f}" for x in probe.squeeze(-1).tolist()))
    print("  " + " ".join(f"{c:>4}" for c in choices))
    finish_run(run_dir, {"baseline_accuracy": baseline_accuracy, "before": before, "history": history})


if __name__ == "__main__":
    main()
```

```bash
python -m scripts.ch06_train_band
```

Observed output (a few seconds on the test machine):

```text
Run: runs/ch06-band/20261002-213906
Baseline 'always answer 0': val accuracy 75.6%
Untrained model: val loss 0.559, val accuracy 75.6%
epoch | train loss | val loss | val acc
    1 |     0.5266 |   0.5000 |   75.6%
    5 |     0.1839 |   0.1693 |   94.0%
   10 |     0.0933 |   0.0975 |   96.8%
   15 |     0.0712 |   0.0714 |   99.6%
   20 |     0.0609 |   0.0613 |   98.8%
   25 |     0.0543 |   0.0667 |   97.2%
   30 |     0.0480 |   0.0526 |   99.4%
Learned answer by input (1 = inside band):
  0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 3.75 4.00
     0    0    0    0    0    0    0    1    1    1    0    0    0    0    0    0    0
```

Read it in order:

1. **Baselines first.** About three quarters of inputs fall outside the band, so the rule "always answer 0" scores 75.6%. The untrained network does exactly as well, by answering 0 for everything. Any trained model must beat this baseline to have learned anything, and the comparison is always worth printing (Chapter 4's discipline).
2. **The loss falls; accuracy rises.** Training and validation loss fall together, and validation accuracy climbs past 99%. Training and validation losses stay close, so there is no sign of overfitting: the network learned the general rule, not the specific examples.
3. **Validation accuracy wobbles** (99.6%, 98.8%, 97.2%, 99.4%) while the loss keeps falling. Inputs very close to the band's edges are genuinely hard, and small parameter changes flip a few of them. This is normal; judge training by trends, not single epochs.
4. **The learned answers trace the band.** Asked across the input range, the network answers 1 from 1.75 to 2.25 and 0 elsewhere. The edge points 1.5 and 2.5 come out as 0: the learned boundaries sit slightly inside the true ones, a small error at the hardest inputs. Nobody told the network where the band was. It found weights that do what Chapter 5's hand-built detector did.

> **A practical note on CPU threads.** This run used several CPU cores at once, as PyTorch does by default, but on a model this small the coordination costs more than it saves. On the test machine, limiting PyTorch to one thread (setting the environment variable `OMP_NUM_THREADS=1` and calling `torch.set_num_threads(1)`) cut the run from about 6.3 seconds to 3.5 seconds. For the larger models in later chapters, multiple threads help. Measure rather than assume.

---

### 6.8 Gradients add up: `zero_grad`, accidental accumulation, and intentional accumulation

`backward()` does not *replace* the contents of `.grad`. It **adds** to them. The reason is a legitimate use (accumulation, below), and the consequence is the most common bug in hand-written training loops.

File: [`code/examples/ch06/accumulation.py`](../../code/examples/ch06/accumulation.py)

```python
"""Chapter 6.8: gradients add up. Accidental versus intentional accumulation.

Run from `code/`:  python examples/ch06/accumulation.py
"""

import torch
from torch import nn

from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data

loss_fn = nn.CrossEntropyLoss()
inputs, targets = make_band_data(64, seed=0)

# 1. backward() ADDS to .grad; it does not replace it.
torch.manual_seed(0)
model = TinyMLP(1, 8, 2)
loss_fn(model(inputs), targets).backward()
first = model.output.bias.grad.clone()
loss_fn(model(inputs), targets).backward()      # same batch again, no zero_grad
print("grad after one backward :", [round(v, 4) for v in first.tolist()])
print("grad after two backwards:", [round(v, 4) for v in model.output.bias.grad.tolist()], "(doubled)")

# 2. Intentional accumulation: 4 micro-batches of 16 give the same gradient as one batch of 64,
#    IF each micro-batch's loss is divided by the number of micro-batches.
torch.manual_seed(0)
full = TinyMLP(1, 8, 2)
torch.manual_seed(0)
accumulated = TinyMLP(1, 8, 2)

loss_fn(full(inputs), targets).backward()
micro_batches = 4
for chunk_inputs, chunk_targets in zip(inputs.chunk(micro_batches), targets.chunk(micro_batches)):
    (loss_fn(accumulated(chunk_inputs), chunk_targets) / micro_batches).backward()

same = all(torch.allclose(a.grad, b.grad, atol=1e-6) for a, b in zip(full.parameters(), accumulated.parameters()))
print("\n4 micro-batches (loss divided by 4) match one full batch:", same)

# Without dividing, the accumulated gradient is 4 times too large.
torch.manual_seed(0)
undivided = TinyMLP(1, 8, 2)
for chunk_inputs, chunk_targets in zip(inputs.chunk(micro_batches), targets.chunk(micro_batches)):
    loss_fn(undivided(chunk_inputs), chunk_targets).backward()
ratio = (undivided.output.bias.grad / full.output.bias.grad).tolist()
print("without dividing, gradient / full-batch gradient:", [round(v, 3) for v in ratio])
```

Observed output:

```text
grad after one backward : [-0.2672, 0.2672]
grad after two backwards: [-0.5343, 0.5343] (doubled)

4 micro-batches (loss divided by 4) match one full batch: True
without dividing, gradient / full-batch gradient: [4.0, 4.0]
```

**Accidental accumulation.** Two backward passes without clearing in between gave exactly double the gradient. In a training loop that forgets `optimizer.zero_grad()`, every step's gradient includes the sum of all previous steps' gradients. The optimizer then steps according to stale, ever-growing information. Exercise 3's solution measures the damage on the band task:

```text
zero_grad=True   val accuracy at epochs 5,10,...,30: 94.0%, 96.8%, 99.6%, 98.8%, 97.2%, 99.4%
zero_grad=False  val accuracy at epochs 5,10,...,30: 25.8%, 75.6%, 41.8%, 75.6%, 75.6%, 75.6%
```

Without `zero_grad`, training never gets past the baseline. It even falls to 25.8% at epoch 5, worse than always answering 0, before ending at exactly the "always answer 0" score. No error message appears. The only symptom is that training does not work, which is why `zero_grad` belongs at a fixed place in every step and why `train_step` puts it first. (Some code calls `zero_grad` after `optimizer.step()` instead; either placement works, as long as it happens once per update.)

**Intentional accumulation.** Sometimes you want the gradient of a large batch, but a large batch does not fit in memory (Chapter 3.12, and Chapter 19.8 at scale). **Gradient accumulation** splits the batch into micro-batches, runs forward and backward on each without clearing, letting the gradients add up, and only then calls `optimizer.step()` once. The example shows the result matches a full batch exactly, **if each micro-batch's loss is divided by the number of micro-batches**. Without dividing, the accumulated gradient is four times too large, which acts like a four-times-larger learning rate.

The difference between the two kinds of accumulation is not the mechanism, which is identical, but intent and bookkeeping: intentional accumulation clears once per *update*, scales each loss correctly, and steps once per *group* of micro-batches. Exercise 4 builds a correct accumulated step; its solution weights each chunk by its share of the examples, so it also works when the batch does not divide evenly.

---

### 6.9 Training mode, evaluation mode, and turning off gradient tracking

Two different switches are easily confused. One changes *what some layers compute*. The other changes *whether PyTorch records computations for a backward pass*.

File: [`code/examples/ch06/modes.py`](../../code/examples/ch06/modes.py)

```python
"""Chapter 6.9: training mode, evaluation mode, and switching off gradient tracking.

Run from `code/`:  python examples/ch06/modes.py
"""

import torch
from torch import nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(8, 2))
x = torch.ones(1, 4)

model.train()
print("train mode, same input three times:", [[round(v, 3) for v in model(x)[0].tolist()] for _ in range(3)])
model.eval()
print("eval mode, same input three times :", [[round(v, 3) for v in model(x)[0].tolist()] for _ in range(3)])

# Dropout in training mode zeroes random values (here half) and scales up the rest.
dropout = nn.Dropout(p=0.5)
dropout.train()
print("\ndropout, training mode:", dropout(torch.ones(10)).tolist())
dropout.eval()
print("dropout, eval mode    :", dropout(torch.ones(10)).tolist())

# Gradient tracking: on by default for computations involving parameters.
model.eval()
tracked = model(x)
with torch.no_grad():
    untracked = model(x)
with torch.inference_mode():
    inference = model(x)
print("\nrecorded for backward? normal:", tracked.requires_grad,
      "| no_grad:", untracked.requires_grad, "| inference_mode:", inference.requires_grad)

# eval() and no_grad() are independent: one changes layer behavior, the other bookkeeping.
model.train()
with torch.no_grad():
    outputs = [round(model(x)[0, 0].item(), 3) for _ in range(3)]
print("no_grad but still train mode (dropout active):", outputs)
```

Observed output:

```text
train mode, same input three times: [[-0.407, -0.15], [0.148, -0.187], [0.164, -0.203]]
eval mode, same input three times : [[-0.139, -0.178], [-0.139, -0.178], [-0.139, -0.178]]

dropout, training mode: [0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0]
dropout, eval mode    : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

recorded for backward? normal: True | no_grad: False | inference_mode: False
no_grad but still train mode (dropout active): [-0.407, -0.345, -0.443]
```

#### Switch 1: `model.train()` and `model.eval()`

Some layers behave differently during training. The main one in this book is **dropout**: in training mode, it sets a random selection of values to zero (here, half of them) and scales up the survivors so the average size is unchanged. This prevents the network from relying too heavily on any single internal value, which tends to reduce overfitting (Chapter 15.6 covers it in transformers). In evaluation mode, dropout passes everything through unchanged.

The output shows the consequence: in training mode, the same input gives three different outputs; in evaluation mode, the same output every time. **If you evaluate without calling `model.eval()`, your validation measurements include random dropout noise and are systematically wrong.** `model.train()` and `model.eval()` set a flag on the module and all its submodules; `evaluate` in `training_basics.py` switches to evaluation mode and restores the caller's mode afterwards.

#### Switch 2: `torch.no_grad()` and `torch.inference_mode()`

During evaluation and generation, you never call `backward()`, so recording the computation wastes memory and time. Inside `with torch.no_grad():`, PyTorch records nothing: results have `requires_grad=False`. `torch.inference_mode()` is a stricter, slightly faster variant for code that will never need gradients at all. `evaluate` uses `@torch.no_grad()` as a decorator, which applies it to the whole function.

#### They are independent

The last line of the output shows `no_grad` with the model still in training mode: dropout is still active and outputs still vary. `no_grad` does not switch layers to evaluation behavior, and `eval()` does not stop recording. Evaluation needs both:

| | Records for backward | Dropout active |
|---|---|---|
| Training step | yes | yes (`train()`) |
| Evaluation / generation | no (`no_grad`) | no (`eval()`) |
| Common bug: `no_grad` only | no | **yes**: noisy, biased measurements |
| Common bug: `eval()` only | **yes**: wasted memory, slower | no |

---

### 6.10 Experiments: learning rates that are too small, too large, and workable

The learning rate is the most important training setting. Here is the band task trained for 30 epochs with each of several learning rates, every run starting from identical parameters:

File: [`code/scripts/ch06_learning_rates.py`](../../code/scripts/ch06_learning_rates.py)

```python
"""Chapter 6.10: the same training run with different learning rates.

Run from `code/`:
    python -m scripts.ch06_learning_rates
    python -m scripts.ch06_learning_rates --optimizer adamw --rates 0.0001 0.001 0.01 0.1 1
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from llmfp.experiment import set_seed
from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data
from llmfp.training_basics import evaluate, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_data = make_band_data(2000, seed=args.seed)
    validation_data = make_band_data(500, seed=args.seed + 1)
    loss_fn = nn.CrossEntropyLoss()
    print(f"optimizer={args.optimizer}, epochs={args.epochs}")
    print(f"{'learning rate':>13} | {'loss after 1 epoch':>18} | {'final train loss':>16} | {'val acc':>7}")
    for rate in args.rates:
        set_seed(args.seed)  # identical starting parameters for every rate
        model = TinyMLP(1, 16, 2)
        optimizer_class = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.AdamW
        optimizer = optimizer_class(model.parameters(), lr=rate)
        history = fit(model, train_data, validation_data, loss_fn, optimizer, args.epochs, 32, args.seed)
        final = history[-1]
        accuracy = evaluate(model, *validation_data, loss_fn)["accuracy"] if math.isfinite(final["train_loss"]) else float("nan")
        note = "" if math.isfinite(final["train_loss"]) else f"  (stopped at epoch {final['epoch']}: loss is not a number)"
        print(f"{rate:>13g} | {history[0]['train_loss']:>18.4f} | {final['train_loss']:>16.4f} | {accuracy:>7.1%}{note}")


if __name__ == "__main__":
    main()
```

With SGD:

```text
optimizer=sgd, epochs=30
learning rate | loss after 1 epoch | final train loss | val acc
        0.001 |             0.5618 |           0.5473 |   75.6%
         0.01 |             0.5602 |           0.4626 |   75.6%
          0.1 |             0.5438 |           0.0979 |   96.0%
            1 |             0.5318 |           0.4301 |   75.6%
           10 |             1.5211 |           1.2043 |   75.6%
          100 |            42.7928 |          17.6136 |   24.4%
```

With AdamW:

```text
optimizer=adamw, epochs=30
learning rate | loss after 1 epoch | final train loss | val acc
       0.0001 |             0.5615 |           0.5347 |   75.6%
        0.001 |             0.5577 |           0.2848 |   88.2%
         0.01 |             0.5266 |           0.0480 |   99.4%
          0.1 |             0.3683 |           0.1487 |  100.0%
            1 |             0.6389 |           0.5754 |   75.6%
           10 |             7.1520 |           0.7595 |   75.6%
```

The pattern, which you will see again in every training run in this book:

- **Too small:** the loss barely moves from its starting value, and accuracy stays at the baseline (75.6%). Nothing is broken; training is just far too slow. This is easy to mistake for "the model cannot learn this".
- **Workable:** a range of rates in the middle all train well (SGD around 0.1; AdamW from about 0.01 to 0.1).
- **Too large:** the loss stays high or gets worse, and accuracy falls back to the baseline or below (SGD at 100 scores 24.4%, below the baseline: it answers 1 for everything). Here the loss became very large but stayed a finite number. On other problems, as in section 6.6, a too-large learning rate drives values to infinity and NaN; `fit` stops and reports it when that happens.
- **The workable range differs by optimizer.** In these sweeps, SGD trained well only at 0.1, while AdamW trained well at both 0.01 and 0.1 and made partial progress at 0.001. AdamW sizes its steps by the learning rate rather than by the gradient, which tends to make it less sensitive to the exact choice. A learning rate is only meaningful together with its optimizer, model, and batch size.

How do practitioners choose? They start from values known to work for similar models and optimizers (AdamW rates for transformers commonly fall between about 0.0001 and 0.001, and published model configurations state theirs), then run short sweeps like this one, on a log scale (each rate ten times the previous), and pick from the stable range. Chapter 19.6 adds learning-rate schedules, which change the rate during training.

---

### 6.11 Common mistakes, recap, concept checks, exercises, answers, and checkpoint

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| Training "runs" but accuracy stays at the baseline | Learning rate far too small or too large; missing `zero_grad`; missing `optimizer.step()` | Check the five lines of `train_step`; run a learning-rate sweep |
| Loss suspiciously low from the start, or does not decrease as expected | Softmax applied before `CrossEntropyLoss` | Pass logits, not shares |
| Loss becomes `inf` or `nan` | Learning rate too large; bad data | Lower the learning rate; check inputs for NaN; Chapter 20.5 |
| `RuntimeError: Trying to backward through the graph a second time` | Reusing a loss or an intermediate result across steps | Recompute the forward pass each step; `detach()` values kept across steps |
| Validation numbers vary between identical runs of `evaluate` | Model left in training mode (dropout active) | `model.eval()` before evaluation |
| Memory grows every step | Storing loss tensors (with their records) instead of `loss.item()` | Store `.item()` or `.detach()` |
| Accumulated training behaves like a larger learning rate | Micro-batch losses not divided | Scale each micro-batch's loss by its share |
| `element 0 of tensors does not require grad and does not have a grad_fn` | Computing the loss inside `no_grad`, or from detached values | Compute training losses with tracking on |
| Results differ between runs despite a seed | Unseeded shuffling | Seeded generator for batches (as in `fit`) |

#### Recap

- **Loss** reduces a model's performance on a batch to one number. **Cross-entropy** is large when the correct candidate gets a small share and punishes confident mistakes hardest; even spread over N candidates gives a fixed reference value.
- A **gradient** reports, for every parameter, the direction that increases the loss and how strongly the loss responds. You can confirm it by nudging a parameter and measuring.
- **Autograd** records operations on marked tensors during the forward pass; `backward()` walks the record backward and stores every gradient in `.grad`, with the parameter's shape.
- An **optimizer** moves parameters against their gradients. **SGD** steps by the learning rate times the gradient. **AdamW** adapts step sizes per parameter, uses momentum, and applies weight decay.
- The **training loop**: train mode, zero gradients, forward, loss, backward, step; repeated over shuffled **minibatches** for several **epochs**, with validation after each.
- `backward()` **adds** to `.grad`. Forgetting `zero_grad` silently ruins training. **Gradient accumulation** uses the same adding on purpose, with scaled losses and one step per group.
- `train()`/`eval()` control layer behavior such as **dropout**; `no_grad`/`inference_mode` control recording. Evaluation needs both.
- The **learning rate** has a workable range: too small barely learns, too large stalls or diverges to `inf`/NaN. The range depends on the optimizer.

#### Concept checks

1. A model gives the correct token a share of 0.9 in one example and 0.01 in another. Which example contributes more to the loss, and why is that useful?
2. A language model with a 50,000-token vocabulary has a validation loss of 10.8 after training. What does that tell you?
3. The gradient for a parameter is -3. What happens to the loss if the parameter is increased slightly? Which way will SGD move it?
4. Why is autograd so much more efficient than nudging each parameter?
5. What does `requires_grad=True` change? Which tensors have it set by default in a network?
6. Why must the SGD update by hand happen inside `torch.no_grad()`?
7. Explain, with the knob example, what happens when the learning rate is too large.
8. Name two ways AdamW differs from SGD, as observed in section 6.6.
9. What is the difference between a minibatch and an epoch?
10. Why does forgetting `zero_grad` not produce an error message?
11. You accumulate gradients over 8 micro-batches but forget to scale the losses. What is the effect?
12. You evaluate a model with dropout inside `torch.no_grad()` but without calling `model.eval()`. What goes wrong?

#### Exercises

**Exercise 1 (read losses).** Without running code, rank these by cross-entropy loss, lowest first, for a correct answer at index 0: logits `[3, 0, 0]`, `[0, 0, 0]`, `[0, 3, 0]`, `[30, 0, 0]`. Then check with `nn.CrossEntropyLoss`. Which of them is the "even spread" reference, and what is its value for 3 candidates?

**Exercise 2 (two knobs).** Extend the crate model to `knob times crates plus offset`, with two parameters. Starting from knob 10 and offset 0, measure by nudging each one separately how the loss responds, and compare with the gradients PyTorch computes for both. Then train both with SGD and report where they end up.

**Exercise 3 (forget `zero_grad`).** Copy the training loop, remove `optimizer.zero_grad()`, and train the band task. Compare validation accuracy over epochs with the correct loop.

**Exercise 4 (accumulate correctly).** Write `accumulated_train_step(model, inputs, targets, loss_fn, optimizer, micro_batches)` that splits the batch into `micro_batches` chunks, accumulates their gradients, and steps once. Write a test showing that after one step, the parameters equal those from a single full-batch step, including when the batch size does not divide evenly.

**Exercise 5 (measure the eval-mode bug).** Build `nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))`, train it on the band task with `fit`, then evaluate the validation set five times with dropout still active (call `model.train()`, then compute the loss inside `torch.no_grad()` yourself, without `evaluate`). Report the five losses and compare them with `evaluate`'s result.

**Exercise 6 (batch size and learning rate).** Run the AdamW learning-rate sweep with batch sizes 8 and 256 (change the batch size in `ch06_learning_rates.py`, or add an option for it). Does the best learning rate change? How does the number of optimizer steps per epoch differ, and how does that help explain what you see?

#### Suggested answers and acceptance criteria

**Concept checks**

1. The 0.01 example, by far: cross-entropy grows steeply as the correct share falls toward zero. Training therefore concentrates on the cases the model gets badly wrong.
2. That is the even-spread reference for 50,000 candidates (10.820). The model has learned essentially nothing; it does no better than spreading its shares evenly. Something is broken.
3. A negative gradient means increasing the parameter decreases the loss. SGD subtracts the learning rate times the gradient, which here means increasing the parameter.
4. Nudging needs one extra evaluation per parameter. Autograd computes every gradient in one backward pass through the recorded computation, at roughly the cost of one forward pass.
5. It tells PyTorch to record operations involving the tensor so that a gradient can be computed for it. Parameters of `nn.Module` layers have it set; input data does not.
6. Otherwise the update would itself be recorded as part of the computation, and the next backward pass would try to work through it.
7. Each step overshoots the best value by more than the last, so the knob swings ever more widely, the loss grows, values eventually overflow to infinity, and then become NaN.
8. Its steps are roughly the size of the learning rate regardless of the gradient's size (adaptive step size), and it keeps moving in its recent average direction, overshooting before turning back (momentum). It also applies weight decay.
9. A minibatch is the subset of examples used for one optimizer step; an epoch is one full pass over all training examples, which consists of many minibatches.
10. Adding to `.grad` is valid, intended behavior (it is how accumulation works), so PyTorch cannot know you forgot to clear it. The only symptom is that training does not improve.
11. The accumulated gradient is 8 times larger than intended, which acts like an 8-times-larger learning rate, possibly enough to destabilize training.
12. Dropout is still active, so each evaluation randomly zeroes values: measurements are noisy and systematically worse than the model's real performance. `no_grad` only stops recording.

**Exercise 1.** `[30, 0, 0]` (nearly 0), `[3, 0, 0]` (about 0.09), `[0, 0, 0]` (1.099, the even-spread reference for 3 candidates), `[0, 3, 0]` (about 3.1). Acceptance: your ranking matches, and you can say why the last is worst (confidently wrong).

**Exercise 2.** Acceptance: for each parameter, the nudged measurement matches PyTorch's gradient to about two decimal places, as in section 6.3; after training with a workable learning rate (find one by trying a few), the knob ends near 12 and the offset near 0, because the data has no fixed extra weight. Note that the offset's gradient is much smaller than the knob's, so with SGD it moves slowly: a small instance of the problem AdamW's adaptive step sizes address.

**Exercise 3.** Solution: [`code/solutions/ch06_forgot_zero_grad.py`](../../code/solutions/ch06_forgot_zero_grad.py); observed output is in section 6.8. Acceptance: the correct loop reaches about 99% validation accuracy; the broken loop does not get past the 75.6% baseline.

**Exercise 4.** Solution: [`code/solutions/ch06_accumulated_step.py`](../../code/solutions/ch06_accumulated_step.py), tests in [`code/tests/test_training_basics.py`](../../code/tests/test_training_basics.py) (`test_accumulated_step_matches_full_batch_step`, run with 2, 3, and 5 micro-batches on 31 examples, so every split is uneven). Observed output:

```text
micro_batches=1: loss before update 0.655976
micro_batches=4: loss before update 0.655976
micro_batches=7: loss before update 0.655976
parameters after one step with 4 micro-batches match 1 full batch: True
parameters after one step with 7 micro-batches match 1 full batch: True
```

Acceptance: `zero_grad` once before the chunks, `step` once after; each chunk's loss weighted by its share of the examples (dividing by the number of chunks is only exact when chunks are equal); the test passes for uneven chunks.

**Exercise 5.** Acceptance: the five dropout-active losses differ from each other and are higher than `evaluate`'s loss, which is repeatable. You can explain that `evaluate` switched to evaluation mode and that dropout zeroes half of the hidden values at random, which makes outputs worse on average.

**Exercise 6.** Acceptance: a table of final train loss and validation accuracy for both batch sizes across the same learning rates, and an explanation. With 2,000 training examples, batch size 8 gives 250 optimizer steps per epoch and batch size 256 gives 8, so over 30 epochs the small-batch run takes about 30 times as many steps. Expect the small batch to train well even at lower learning rates (many steps), and the large batch to need a higher rate or more epochs. Batch size and learning rate interact. Chapter 19.8 returns to this at scale.

#### Checkpoint: what you can now do independently

You can now:

- Explain training as measure, compute gradients, step, repeat, without formulas, and point to the line of code for each part.
- Read cross-entropy values against reference points and recognize the softmax-before-loss bug.
- Use autograd, and confirm a gradient by nudging.
- Write a training loop with seeded minibatches, epochs, validation, baselines, and run records.
- Avoid accidental gradient accumulation, and implement intentional accumulation correctly.
- Use `train()`/`eval()` and `no_grad()` correctly and know which problem each solves.
- Diagnose learning rates that are too small or too large, and run a sweep to find a workable one.

**Next:** Chapter 7 puts it all together: a network trained on harbor text to predict the next character, compared head to head with the counting model.
