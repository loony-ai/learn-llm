## Chapter 5: Neural Networks Through Behavior and Code

[Back to index](../../README.md) · Previous: [Chapter 4](ch04-data-experiments-reproducibility.md) · Next: [Chapter 6](ch06-how-training-works.md)

The counting model fails completely on contexts it has never seen (Chapter 1.11), and Chapter 4 measured how quickly that failure grows as contexts get longer. A neural network takes a different approach. Instead of looking up stored counts, it *computes* a score for every candidate from numbers describing the input, using a large set of adjustable numbers. Because it computes rather than looks up, it produces scores for any input at all, including inputs it has never seen.

This chapter explains what a neural network is by building one and watching what it does. There are no formulas. Every piece is a small amount of code whose behavior you can observe: a unit that scores an input, a layer of many units, the activation functions that make stacking layers worthwhile, PyTorch's module system, and the conversion of raw scores into a ranked choice. Training, the process that makes the adjustable numbers useful, is the subject of Chapter 6. Here they are set by hand or left random.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain what a unit (neuron) does with its inputs, weights, and bias, and run one by hand.
2. Describe a linear layer as many units applied at once, and predict its parameter shapes and output shapes.
3. Explain, from observed behavior, why an activation function is needed between linear layers, and describe what ReLU, GELU, tanh, and sigmoid do.
4. Build networks as PyTorch `nn.Module` classes, list and count their parameters, and explain what the forward pass is.
5. Describe logits and what softmax does to them, from observation, and connect it to choosing tokens.
6. Track tensor shapes through a network, and use a shape trace to find where a shape goes wrong.

#### Prerequisites

- [Chapter 1](ch01-what-a-language-model-predicts.md): scores for candidate next tokens, greedy choice and sampling (sections 1.6–1.7).
- [Chapter 2.6](ch02-python-foundations-and-environment.md): classes, inheritance, `super().__init__()`, and the `__call__`/`forward` pattern.
- [Chapter 3](ch03-tensors.md): shapes, axes, dtypes, broadcasting, `argmax`, batches.
- [Chapter 4.6](ch04-data-experiments-reproducibility.md): seeds.

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Neural network | A program that computes outputs from inputs through layers of simple adjustable steps | 5.1 |
| Unit (neuron) | One adjustable scoring rule: multiply each input by a weight, add the results and a bias | 5.2 |
| Weight, bias | The adjustable numbers of a unit: one weight per input, plus one bias | 5.2 |
| Linear layer | Many units reading the same inputs, computed together | 5.3 |
| Matrix multiplication | The tensor operation that computes a whole linear layer for a whole batch at once | 5.3 |
| Activation function | A simple fixed function applied to every value between layers, which lets stacked layers represent bends | 5.4 |
| ReLU, GELU, tanh, sigmoid | Common activation functions | 5.4 |
| Hidden layer, MLP | A layer between input and output / a stack of linear layers with activations between them | 5.4 |
| `nn.Module` | PyTorch's base class for anything with parameters and a forward computation | 5.5 |
| Forward pass | Running a network on an input to get its output | 5.5 |
| Initialization | The random starting values given to parameters | 5.6 |
| Logits | A network's raw output scores, one per candidate, before any conversion | 5.7 |
| Softmax | A function that turns a list of scores into positive shares that total 1, preserving their order | 5.7 |
| Forward hook | A function PyTorch calls after a module runs, used here to record shapes | 5.8 |

---

### 5.1 The problem: a count table cannot handle contexts it never saw

The counting model is a lookup: context in, stored counts out. When a context was never stored, there is nothing to return. Backoff (Exercise 1.5) helped a little by trying shorter contexts, but the model still had no notion that two different contexts might call for similar predictions. "The keeper fixed the" and "the keeper repaired the" share nothing as far as a count table is concerned.

What would a better approach need?

1. **A way to describe a context with numbers**, so that similar contexts get similar descriptions. Chapter 7 starts with a simple description, and Chapter 10 introduces learned descriptions called embeddings.
2. **A function that turns those numbers into a score for every candidate token**, for any description, including ones never seen before.
3. **Adjustable numbers inside that function**, so that a training process can tune it until its scores match the data (Chapter 6).

A **neural network** is that function. It is built from very simple steps, each with adjustable numbers, arranged in layers. The name comes from a loose historical analogy with neurons in the brain. **The analogy is not a description of how the software works**, and nothing in this book depends on it: a neural network is ordinary code that multiplies, adds, and applies simple functions to tensors.

---

### 5.2 A single unit: an adjustable scoring rule

The smallest building block is the **unit** (also called a *neuron*). It takes a list of input numbers and produces one output number. It has its own adjustable numbers: one **weight** for each input, and one **bias**. It multiplies each input by that input's weight, adds up the results, and adds the bias.

Here is one, written as a plain Python function, scoring how urgent a harbor message is from three yes/no features (1 means yes, 0 means no):

File: [`code/examples/ch05/single_unit.py`](../../code/examples/ch05/single_unit.py)

```python
@@FILE code/examples/ch05/single_unit.py@@
```

Observed output:

```text
@@RUN python examples/ch05/single_unit.py@@
```

The same code with different weights produces completely different behavior:

- A large positive weight means "this input pushes the score up". With weights 2 and 3 on storm and injury, the message mentioning both scores highest.
- A negative weight means "this input pushes the score down". The market weight of -1 lowers the market message's score.
- A zero weight means "ignore this input". The second setting ignores everything except the market.
- The bias shifts every score by the same amount, regardless of the inputs.

This is the essential idea of a neural network in miniature: **a fixed procedure whose behavior is decided by adjustable numbers**. The procedure is the architecture; the weights and biases are the parameters (Chapter 1.4). In this example a person chose them. In a real network there are thousands to billions of them, far too many to set by hand, and training sets them from data.

A single unit can only express "add up evidence, each piece with its own importance". The rest of this chapter shows how combining many units lets a network express much more.

---

### 5.3 Linear layers: many scoring rules at once

A **linear layer** is a group of units that all read the same inputs. Each unit has its own weights and bias, so each computes a different score from the same inputs. In PyTorch it is `nn.Linear(in_features, out_features)`: `in_features` inputs, `out_features` units, and so `out_features` outputs.

File: [`code/examples/ch05/linear_layer.py`](../../code/examples/ch05/linear_layer.py)

```python
@@FILE code/examples/ch05/linear_layer.py@@
```

Observed output:

```text
@@RUN python examples/ch05/linear_layer.py@@
```

What to take from it:

- **Parameter shapes.** `weight` has shape `(out_features, in_features)`: one row of weights per unit. `bias` has shape `(out_features,)`. A layer from 3 inputs to 2 outputs has 6 weights and 2 biases. Notice the order: PyTorch stores *outputs first*, which surprises many people and matters when loading weights saved by other software (Chapter 22.4).
- **The layer computes every unit for every example in the batch.** Input `(2, 3)` (2 examples, 3 features) gave output `(2, 2)` (2 examples, 2 scores). The triple loop computed exactly the same numbers one multiplication at a time.
- **Matrix multiplication.** The operation `x @ layer.weight.T + layer.bias` reproduces the layer. `@` is PyTorch's **matrix multiplication** operator. Operationally, for every example (row of `x`) and every unit (row of the weight tensor), it multiplies matching positions and adds up the products: the inner part of the triple loop, for all pairs at once, in optimized compiled code. `.T` flips the weight tensor so that its rows line up correctly, and the bias is added to every example by broadcasting (Chapter 3.7). You will rarely write this yourself; `nn.Linear` does it. But matrix multiplication is the operation that dominates the running time of neural networks, which is why GPUs, which are built to do it fast, matter so much (Chapter 3.11).
- **Only the last axis is transformed.** A `(4, 7, 3)` input, meaning (batch, sequence, features), became `(4, 7, 2)`. The layer is applied independently to every position of every sequence. Transformers rely on this constantly: the same layer processes every token.

The name "linear" describes the layer's behavior, which the next section makes visible.

---

### 5.4 Activation functions: what stacking layers needs

If one layer computes several scores from the inputs, it seems natural to feed those scores into another layer, and another, building up more complicated behavior. The first experiment shows that, on its own, this does not work.

File: [`code/examples/ch05/why_activations.py`](../../code/examples/ch05/why_activations.py)

```python
@@FILE code/examples/ch05/why_activations.py@@
```

The network class used here is the book's `TinyMLP`, defined in [`code/llmfp/nn_basics.py`](../../code/llmfp/nn_basics.py) and listed in full in section 5.5: a linear layer, then an activation function (or nothing), then another linear layer.

Observed output:

```text
@@RUN python examples/ch05/why_activations.py@@
```

**Part 1: two linear layers with nothing between them.** Feed the network evenly spaced inputs (0, 1, 2, 3, 4) and the outputs are also evenly spaced: every step between neighbors is identical. The network responds "in a straight line": change the input by a fixed amount and the output changes by a fixed amount, everywhere. That is what *linear* means here. Worse, the two layers can be merged into a single equivalent linear layer, as the script confirms. Stacking linear layers adds parameters but no new abilities: a hundred stacked linear layers behave like one.

**Part 2: the same network with ReLU between the layers.** Now the steps between neighbors differ: the response can bend.

**Part 3: what bending makes possible.** With three hand-set hidden units and ReLU, the network outputs 1 for an input of 2, 0.5 at 1.5 and 2.5, and 0 everywhere else: a *band detector*. No linear layer can do this, because a straight-line response can only rise steadily, fall steadily, or stay flat; it cannot rise and then fall again. Each hidden unit stays at zero until the input passes a certain point and then rises; the output layer combines three such units, adding two and subtracting one twice, to build a peak. Exercise 6 extends this to two peaks.

That is the role of an **activation function**: a simple, fixed function applied to every value between layers, with no adjustable numbers of its own. It is what lets stacked layers build up bends, peaks, thresholds, and combinations of them. With enough hidden units and suitable parameter values, networks of this shape can approximate an extremely wide range of input-to-output behaviors. That is a known theoretical result, stated here without its mathematical proof. Note what it does *not* say: it does not promise that training will find those parameter values, or that a network of practical size is enough for a given task. Those are empirical questions.

#### Common activation functions

File: [`code/examples/ch05/activations_tour.py`](../../code/examples/ch05/activations_tour.py)

```python
@@FILE code/examples/ch05/activations_tour.py@@
```

Observed output:

```text
@@RUN python examples/ch05/activations_tour.py@@
```

| Activation | Behavior you can see in the table | Where it appears |
|---|---|---|
| **ReLU** (rectified linear unit) | Negative values become 0; positive values pass unchanged | Simple, fast, widely used; this book's small networks |
| **GELU** | Like ReLU, but with a smooth curve near zero; small negative values give small negative outputs | GPT-2 and many transformers (Chapter 15.2) |
| **tanh** | Squeezes every value into the range -1 to 1 | Older networks; some gating mechanisms |
| **sigmoid** | Squeezes every value into the range 0 to 1 | Gates, and yes/no outputs |

Why so many? They trade off simplicity, speed, and how well training works with them in practice. The choice is largely an engineering convention established by what has worked in published models; Chapter 15.2 covers the variants transformers use.

#### Hidden layers and MLPs

A layer between the input and the output is a **hidden layer**: its outputs are intermediate values that nobody looks at directly. A stack of linear layers with activations between them is called a **multi-layer perceptron (MLP)**, a historical name. `TinyMLP` has one hidden layer. Deep learning (Chapter 1.2) means networks with many layers; a transformer contains an MLP inside every one of its blocks (Chapter 15).

---

### 5.5 `nn.Module`, parameters, and the forward pass

PyTorch organizes networks as classes that inherit from `nn.Module`. This section shows the rules, which are few but strict.

File: [`code/examples/ch05/module_basics.py`](../../code/examples/ch05/module_basics.py)

```python
@@FILE code/examples/ch05/module_basics.py@@
```

Observed output:

```text
@@RUN python examples/ch05/module_basics.py@@
```

The rules, and why each exists:

1. **Call `super().__init__()` first in `__init__`.** It sets up the machinery that tracks parameters and submodules. The last part of the output shows that forgetting it fails immediately, the moment you assign a layer.
2. **Assign layers as attributes in `__init__`.** When you write `self.hidden = nn.Linear(3, 4)`, the module records `hidden` as a submodule, and its weight and bias as parameters named `hidden.weight` and `hidden.bias`. A plain value such as `self.scale = 2.0` is just an attribute, not a parameter: training will never adjust it.
3. **Write the computation in `forward`.** The **forward pass** is the computation from input to output. You write it in `forward`, but you run it by calling the module: `model(batch)`. As Chapter 2.6 explained, calling the object runs `__call__`, which does PyTorch's bookkeeping (including the hooks used in section 5.8) and then calls your `forward`. **Always call the module, not `forward` directly**, or that bookkeeping is skipped.
4. **Modules nest.** A module can contain modules that contain modules. `named_parameters()` walks the whole tree and names every parameter with its path. A transformer is a module containing a list of block modules, each containing attention and MLP modules (Part 3).
5. **`nn.Sequential`** chains modules in order without a custom class, convenient for simple stacks.

Each parameter also reports `requires_grad=True`: it is marked as something training may adjust. Chapter 6 explains what that flag does.

Here is the book's reusable module for this chapter:

File: [`code/llmfp/nn_basics.py`](../../code/llmfp/nn_basics.py)

```python
@@FILE code/llmfp/nn_basics.py@@
```

`TinyMLP` is the network from section 5.4. The other functions inspect networks: `count_parameters` adds up the sizes of all parameter tensors, `parameter_table` and `format_parameter_table` list them, and `shape_trace` is explained in section 5.8.

---

### 5.6 Counting and inspecting parameters

How many parameters does a network have? Add up the sizes of its parameter tensors. For a linear layer, that is one weight per (output, input) pair plus one bias per output. The milestone script in section 5.8 prints this table for a small network that scores the 82 words of the harbor vocabulary:

```text
@@RUN python -m scripts.ch05_untrained_network | sed -n '/^parameter/,/^Parameter memory/p'@@
```

Two practical points:

- **The output layer is often the largest.** Here it has 82 rows, one per vocabulary word. Language models have vocabularies of tens of thousands of tokens, so their output layer alone can hold tens of millions of parameters. Chapter 16.5 shows a common trick for sharing it with the input side (weight tying).
- **Memory follows from the count.** 3,250 float32 parameters take 12.7 KiB. The same arithmetic in Chapter 3.12 gave 473 MiB for 124 million parameters.

#### Initialization

Where do the starting values come from? When `nn.Linear` is created, PyTorch fills its weights and biases with small random numbers. This is **initialization**. The output in section 5.5 showed that two networks with the same architecture but different seeds start with different parameters, and so give different scores for the same input; the same seed gives identical parameters, which `tests/test_nn_basics.py` checks.

Why random rather than, say, all zeros? If every unit in a layer started with identical weights, every unit would compute the same output and, as you will see in Chapter 6, receive the same adjustments during training, so they would stay identical forever: a layer of a hundred units would behave like one. Random starting values break that symmetry. The *scale* of the random values also matters for training to work well in deep networks; PyTorch's defaults are sensible for small networks, and Chapter 16.4 examines initialization for transformers.

---

### 5.7 Scores (logits) and turning them into a ranked choice

A network that predicts the next token ends with a linear layer with one output per vocabulary token. Those raw outputs are called **logits**. They are arbitrary numbers: positive or negative, large or small. Only their order and their differences carry meaning: a higher logit means "more likely to come next, according to the network".

The counting model's scores were counts, which you could read directly. To sample from logits, or to compare them with the actual next token during training (Chapter 6.2), we convert them into positive shares that total 1. The function that does this is **softmax**. Rather than define it with a formula, observe what it does:

File: [`code/examples/ch05/softmax_behavior.py`](../../code/examples/ch05/softmax_behavior.py)

```python
@@FILE code/examples/ch05/softmax_behavior.py@@
```

Observed output:

```text
@@RUN python examples/ch05/softmax_behavior.py@@
```

What softmax does, from the output:

- **Every share is positive and they total 1.** The result can be treated as "how strongly the network favors each candidate", and sampled from like Chapter 1's tickets in a hat. The 1,000 samples landed roughly in proportion to the shares.
- **Order is preserved.** The highest logit gets the largest share, so `argmax` gives the same answer before and after softmax. Greedy choice does not need softmax at all.
- **Only differences between logits matter.** Adding 10 to every logit, or subtracting 5, changed nothing. This is why logits have no fixed meaning on their own.
- **Larger gaps make the shares more lopsided.** Doubling the gaps pushed "lit" from 0.644 to 0.865; halving them made the shares more even. Equal logits give equal shares. One logit far above the rest takes nearly everything. Chapter 21.4 uses exactly this behavior: *temperature* is a setting that divides the logits by a number before softmax, to make sampling more or less adventurous.
- **The axis matters.** On a batch, softmax must run along the candidates axis (`dim=-1`). Run along the wrong axis, it produces shares that total 1 across the *batch* instead, which is meaningless and raises no error.

The word *softmax* describes the behavior: a "soft" version of picking the maximum, which gives most of the weight to the largest value without discarding the others entirely.

> **Shares are not probabilities of being correct.** It is common to call softmax outputs "probabilities", and in a technical sense they behave like them. But a share of 0.9 for a token means the network strongly favors it, not that it is right 90% of the time. Whether a model's shares match how often it is actually right is a separate, measurable property called *calibration*, discussed in Chapter 38.4. Untrained or poorly trained networks can be confidently wrong.

---

### 5.8 A network as a function from tensors to tensors: tracking shapes

Every network in this book is a function from tensors to tensors. Reasoning about it starts with shapes: what goes in, what comes out, and what every layer in between produces.

#### Shape traces with forward hooks

`shape_trace` in `nn_basics.py` records the output shape of every submodule during one forward pass. It uses **forward hooks**: functions you register on a module, which PyTorch calls after that module's forward computation finishes (this is part of the bookkeeping `__call__` does, section 5.5). The hook appends the module's name and output shape to a list. The `try`/`finally` removes the hooks even if the forward pass fails, so tracing never leaves a model altered; a test checks this.

#### Milestone: an untrained network that scores every harbor word

File: [`code/scripts/ch05_untrained_network.py`](../../code/scripts/ch05_untrained_network.py)

```python
@@FILE code/scripts/ch05_untrained_network.py@@
```

```bash
python -m scripts.ch05_untrained_network
```

Observed output:

```text
@@RUN python -m scripts.ch05_untrained_network@@
```

And with a different seed:

```text
@@RUN python -m scripts.ch05_untrained_network --seed 1 | tail -7@@
```

Read the shape trace from top to bottom: four contexts of 16 numbers each become four lists of 32 hidden values, stay that shape through ReLU (activations change values, never shapes), and become four lists of 82 logits, one per vocabulary word. Every context, whatever its numbers, gets a score for every word. The counting model's "unseen context" failure is gone.

But look at the top five words. They are arbitrary: different seeds rank completely different words first, and the shares are barely above an even split (about 0.012 per word for 82 words). An untrained network is a working function with meaningless outputs. Making them meaningful is the job of training, which Chapter 6 explains, and Chapter 7 then applies to real harbor text.

The line `with torch.no_grad():` tells PyTorch not to prepare for training during this forward pass. Chapter 6.9 explains what that means and why it matters.

#### Reading a shape error

Give `TinyMLP(16, 32, 82)` inputs with 15 features instead of 16, and the forward pass fails with:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x15 and 16x32)
```

Decode it: `mat1` is the input, 4 examples by 15 features. `mat2` is the first layer's weight, arranged for 16 inputs and 32 outputs. The middle numbers, 15 and 16, must match and do not. The fix is in whichever side is wrong: the data pipeline producing 15 features, or the layer expecting 16. When the failing layer is deep inside a large model, `shape_trace` with a smaller input, or the habits from Chapter 3.13, show which layer received what.

---

### 5.9 Common mistakes, recap, concept checks, exercises, answers, and checkpoint

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| `cannot assign module before Module.__init__() call` | Missing `super().__init__()` | Call it first in `__init__` |
| A layer's parameters are missing from `named_parameters()` and never train | Layers stored in a plain Python list or dict | Use `nn.ModuleList` / `nn.ModuleDict` (used from Chapter 15), or assign as attributes |
| `mat1 and mat2 shapes cannot be multiplied (AxB and CxD)` | Input feature count B differs from the layer's `in_features` C | Fix the data or the layer so B and C match |
| A deep stack of layers behaves no better than one layer | No activation between linear layers | Add an activation |
| Hooks or other features silently skipped | Calling `model.forward(x)` instead of `model(x)` | Call the module |
| Softmax outputs do not total 1 per example | Softmax along the wrong axis | Use `dim=-1` for (…, candidates) tensors |
| Two "identical" runs give different initial outputs | No seed, or seed set after the model was created | `set_seed` before building the model |
| Network outputs used as if they were reliable confidence | Treating softmax shares as measured accuracy | Measure calibration (Chapter 38.4) |

#### Recap

- A **unit** multiplies each input by its own weight, adds the results, and adds a bias. Its behavior is entirely decided by those adjustable numbers.
- A **linear layer** is many units over the same inputs; its weight has shape `(out_features, in_features)`. It transforms only the last axis, for every example and position at once, using **matrix multiplication**.
- Stacked linear layers without activations collapse into one linear layer. **Activation functions** such as ReLU and GELU let networks represent bends, peaks, and thresholds.
- PyTorch networks are `nn.Module` subclasses: call `super().__init__()`, assign layers as attributes, write `forward`, call the module.
- Parameters are counted by adding up tensor sizes; they start from random **initialization**, which a seed makes reproducible.
- **Logits** are raw scores; **softmax** turns them into positive shares totaling 1, preserving order, depending only on differences, and growing more lopsided as gaps grow.
- An untrained network scores every candidate for any input, but its scores are meaningless until training.

#### Concept checks

1. A unit has weights 2, -1, 0 and bias 1. What does it output for inputs 1, 1, 5? Which input is ignored?
2. What are the shapes of the weight and bias of `nn.Linear(768, 3072)`, and how many parameters does it have?
3. What does `nn.Linear(3, 2)` do to an input of shape `(8, 10, 3)`?
4. Why can't two stacked linear layers detect a band of input values, while two linear layers with a ReLU between them can?
5. Why does a network need random rather than identical starting values?
6. What is the difference between `self.layer = nn.Linear(3, 2)` and `self.scale = 2.0` inside a module?
7. Why call `model(x)` rather than `model.forward(x)`?
8. Logits `[1, 2, 3]` and `[101, 102, 103]`: how do their softmax outputs compare? What about `[1, 2, 3]` and `[2, 4, 6]`?
9. Does greedy choice need softmax? Does sampling?
10. A network's softmax share for a word is 0.95. Does that mean the word is correct 95% of the time?
11. Read this error: `mat1 and mat2 shapes cannot be multiplied (32x100 and 128x64)`. What is wrong?
12. Why does the untrained network in section 5.8 give nearly even shares to all 82 words?

#### Exercises

**Exercise 1 (a unit by hand).** Using the `unit` function from `single_unit.py`, find weights and a bias such that a message mentioning only the market scores above every other message in the example, and a message mentioning a storm and an injury scores below zero. Then check the same weights with an `nn.Linear(3, 1)` whose parameters you set with `torch.no_grad()` and `copy_`.

**Exercise 2 (count before you run).** Predict the parameter count of: `nn.Linear(10, 4)`; `TinyMLP(16, 32, 82)`; `TinyMLP(768, 3072, 768)`; `nn.Sequential(nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 1))`. Check each with `count_parameters`.

**Exercise 3 (collapse a stack).** Build `TinyMLP(4, 6, 2, activation="none")` and write code that constructs a single `nn.Linear(4, 2)` producing the same outputs. Write a test that compares them on random inputs. Then show the same construction fails (outputs differ) when the activation is ReLU.

**Exercise 4 (softmax experiments).** Predict, then check: (a) the softmax of `[0, 0, 0, 10]`; (b) what happens to the share of the top candidate as you multiply `[2, 1, 0, -1]` by 0.1, 1, 10, and 100; (c) what happens to the ranking when you add a different constant to each logit. Write one sentence per result explaining it in terms of section 5.7.

**Exercise 5 (find the broken layer).** Write a module with three linear layers where the second layer's `in_features` is wrong by one. Run it, read the error, and use `shape_trace` on the first layer alone (or catch the error inside `shape_trace`) to show which shapes were produced before the failure. Fix it.

**Exercise 6 (two bands).** Hand-set the parameters of a `TinyMLP(1, 6, 1)` with ReLU so it outputs 1 for inputs of 1 and 3, 0.5 at a quarter of the way from each peak to zero, and 0 for inputs below 0.5, between 1.5 and 2.5, and above 3.5. Test it.

#### Suggested answers and acceptance criteria

**Concept checks**

1. Multiply each input by its weight: 2, -1, and 0; add them, giving 1; add the bias of 1: the output is 2. The third input is ignored because its weight is 0.
2. Weight `(3072, 768)`, bias `(3072,)`. Parameters: 3072 rows of 768 weights plus 3072 biases, which `count_parameters` reports as 2,362,368.
3. It applies to the last axis only: output shape `(8, 10, 2)`, the same layer used at every one of the 80 positions.
4. Stacked linear layers merge into one, whose response can only rise steadily, fall steadily, or stay flat. A ReLU between them lets hidden units switch on at different points, and combining them builds a rise followed by a fall.
5. Identical units compute identical outputs and receive identical training adjustments, so they would never become different; the layer would act as a single unit.
6. The layer is registered as a submodule and its weight and bias become trainable parameters. `self.scale` is an ordinary attribute that training never changes.
7. Calling the module runs `__call__`, which performs PyTorch's bookkeeping (such as hooks) before and after `forward`.
8. Identical: adding the same constant to every logit changes nothing. `[2, 4, 6]` has doubled gaps, so its shares are more lopsided toward the top candidate.
9. Greedy choice does not, because `argmax` is unchanged by softmax. Sampling does, because it needs positive shares to choose in proportion to.
10. No. It means the network strongly favors the word. How often such confident choices are correct is a separate property, calibration, which must be measured.
11. The input has 100 features per example (32 examples), but the layer expects 128 inputs. Either the data produces the wrong number of features or the layer was built with the wrong `in_features`.
12. Its parameters are small random numbers, so its logits are small and close together; softmax of nearly equal logits gives nearly equal shares.

**Exercise 1.** One answer: weights `[-2, -2, 3]`, bias `0`. Market only scores 3; storm only and injury only score -2; storm with injury scores -4. Acceptance: your `nn.Linear(3, 1)` with `weight.copy_(torch.tensor([[-2.0, -2.0, 3.0]]))` and `bias.zero_()` gives the same four scores as the `unit` function.

**Exercise 2.** 44; 3,250; 4,722,432; and 66 (30 for each of the two 5-to-5 layers, 6 for the last layer). Acceptance: your predictions match `count_parameters`, and you can explain each as rows times inputs plus one bias per row, for each linear layer, with activations contributing nothing.

**Exercise 3.** The construction is in `test_no_activation_network_is_equivalent_to_one_linear_layer` in [`code/tests/test_nn_basics.py`](../../code/tests/test_nn_basics.py) and in `why_activations.py`: the merged weight is the output weight matrix-multiplied by the hidden weight, and the merged bias is the output weight matrix-multiplied by the hidden bias, plus the output bias. Acceptance: your test passes for `"none"`, and the same construction gives different outputs (`torch.allclose` returns `False`) for `"relu"` on random inputs that include negative hidden values.

**Exercise 4.** (a) The last candidate takes nearly all of the share (observed: 0.99986), and the other three share the tiny remainder equally (about 0.00005 each). (b) Observed top shares: 0.289 at 0.1 (close to an even 0.25), 0.644 at 1 (as in section 5.7), and 1.0 when rounded at 10 and 100, because the gaps grow with the multiplier. (c) Adding *different* constants changes the differences, so it can change the ranking; adding the *same* constant cannot. Acceptance: predictions written before running, and each explanation refers to "only differences matter" or "larger gaps, more lopsided".

**Exercise 5.** Acceptance: the error message names the mismatched sizes (the second layer's expected input count against the first layer's output count); you show the first layer's output shape with `shape_trace(model.first_layer_name, x)` or by catching the error and printing the partial trace; the fixed model's trace shows every layer's shape through to the output.

**Exercise 6.** Solution: [`code/solutions/ch05_two_bands.py`](../../code/solutions/ch05_two_bands.py), tests in [`code/tests/test_ch05_solutions.py`](../../code/tests/test_ch05_solutions.py). Observed output:

```text
@@RUN python -m solutions.ch05_two_bands@@
```

Acceptance: peaks of exactly 1 at 1 and 3, 0.5 halfway down each side, and 0 elsewhere. Note how much hand-design two simple bumps took. Real tasks need thousands of such features, combined in ways nobody could write down, which is why parameters must be learned rather than set (Chapter 6).

#### Checkpoint: what you can now do independently

You can now:

- Explain what a unit, a linear layer, and an activation function each do, using observed behavior rather than formulas.
- Build networks as `nn.Module` classes, follow PyTorch's rules for registering parameters, and run forward passes on batches.
- Predict parameter shapes and counts, and estimate the memory they need.
- Convert logits into shares with softmax, predict how changes to logits affect those shares, and choose tokens greedily or by sampling.
- Trace shapes through a network and decode shape errors.

**Next:** [Chapter 6](ch06-how-training-works.md) explains how training adjusts parameters so that a network's scores match the data: loss, gradients, optimizers, and the training loop.
