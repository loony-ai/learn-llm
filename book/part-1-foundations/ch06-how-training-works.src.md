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
@@FILE code/examples/ch06/cross_entropy.py@@
```

Observed output:

```text
@@RUN python examples/ch06/cross_entropy.py@@
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
@@FILE code/examples/ch06/nudge_one_knob.py@@
```

Observed output:

```text
@@RUN python examples/ch06/nudge_one_knob.py@@
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
@@FILE code/examples/ch06/autograd_basics.py@@
```

Observed output:

```text
@@RUN python examples/ch06/autograd_basics.py@@
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
@@FILE code/examples/ch06/optimizer_steps.py@@
```

Observed output:

```text
@@RUN python examples/ch06/optimizer_steps.py@@
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
@@FILE code/llmfp/training_basics.py@@
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
@@FILE code/llmfp/toy_data.py@@
```

The script, driven by a configuration file and recorded as a run (Chapter 4):

File: [`code/configs/band-cpu.toml`](../../code/configs/band-cpu.toml)

```toml
@@FILE code/configs/band-cpu.toml@@
```

File: [`code/scripts/ch06_train_band.py`](../../code/scripts/ch06_train_band.py)

```python
@@FILE code/scripts/ch06_train_band.py@@
```

```bash
python -m scripts.ch06_train_band
```

Observed output (a few seconds on the test machine):

```text
@@RUN python -m scripts.ch06_train_band@@
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
@@FILE code/examples/ch06/accumulation.py@@
```

Observed output:

```text
@@RUN python examples/ch06/accumulation.py@@
```

**Accidental accumulation.** Two backward passes without clearing in between gave exactly double the gradient. In a training loop that forgets `optimizer.zero_grad()`, every step's gradient includes the sum of all previous steps' gradients. The optimizer then steps according to stale, ever-growing information. Exercise 3's solution measures the damage on the band task:

```text
@@RUN python -m solutions.ch06_forgot_zero_grad@@
```

Without `zero_grad`, training never gets past the baseline. It even falls to 25.8% at epoch 5, worse than always answering 0, before ending at exactly the "always answer 0" score. No error message appears. The only symptom is that training does not work, which is why `zero_grad` belongs at a fixed place in every step and why `train_step` puts it first. (Some code calls `zero_grad` after `optimizer.step()` instead; either placement works, as long as it happens once per update.)

**Intentional accumulation.** Sometimes you want the gradient of a large batch, but a large batch does not fit in memory (Chapter 3.12, and Chapter 19.8 at scale). **Gradient accumulation** splits the batch into micro-batches, runs forward and backward on each without clearing, letting the gradients add up, and only then calls `optimizer.step()` once. The example shows the result matches a full batch exactly, **if each micro-batch's loss is divided by the number of micro-batches**. Without dividing, the accumulated gradient is four times too large, which acts like a four-times-larger learning rate.

The difference between the two kinds of accumulation is not the mechanism, which is identical, but intent and bookkeeping: intentional accumulation clears once per *update*, scales each loss correctly, and steps once per *group* of micro-batches. Exercise 4 builds a correct accumulated step; its solution weights each chunk by its share of the examples, so it also works when the batch does not divide evenly.

---

### 6.9 Training mode, evaluation mode, and turning off gradient tracking

Two different switches are easily confused. One changes *what some layers compute*. The other changes *whether PyTorch records computations for a backward pass*.

File: [`code/examples/ch06/modes.py`](../../code/examples/ch06/modes.py)

```python
@@FILE code/examples/ch06/modes.py@@
```

Observed output:

```text
@@RUN python examples/ch06/modes.py@@
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
@@FILE code/scripts/ch06_learning_rates.py@@
```

With SGD:

```text
@@RUN python -m scripts.ch06_learning_rates@@
```

With AdamW:

```text
@@RUN python -m scripts.ch06_learning_rates --optimizer adamw --rates 0.0001 0.001 0.01 0.1 1 10@@
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
@@RUN python -m solutions.ch06_accumulated_step@@
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
