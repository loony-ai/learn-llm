## Chapter 3: Tensors: NumPy and PyTorch as Containers of Numbers

[Back to index](../../README.md) · Previous: [Chapter 2](ch02-python-foundations-and-environment.md) · Next: Chapter 4 (planned)

The counting model in Chapter 1 stored its parameters in Python dictionaries: a few hundred counts, each with a readable label. A neural network stores millions or billions of unlabeled numbers, and on every prediction it combines huge blocks of them at once. Python dictionaries and lists cannot do that fast enough. The data structure that can is the **tensor**: a block of numbers of a single type, arranged in a grid with a fixed shape.

Almost every bug in model code you will write is, at bottom, a tensor holding the wrong shape, the wrong type, or living on the wrong device. This chapter teaches tensors as an engineer needs them: what they hold, how to look inside them, how to rearrange them, and how to tell when one is not what you think it is. No mathematics is needed or used. A tensor here is a container, and every operation is described by what it does to the numbers inside.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain why numerical code uses arrays instead of Python lists, and observe the speed difference.
2. Read and predict a tensor's shape, number of dimensions, and the meaning of each axis.
3. Choose an appropriate data type and explain what float32, float16, bfloat16, and int64 trade off.
4. Index, slice, mask, reshape, transpose, and batch tensors, and know which operations share memory.
5. Predict the result of broadcasting, and recognize the silent bug it can cause.
6. Reduce tensors along an axis (sum, mean, max, argmax) and keep or drop that axis deliberately.
7. Move data between NumPy and PyTorch, and between CPU and GPU, and estimate memory use from a shape and a data type.
8. Apply a routine for debugging shape errors.

#### Prerequisites

- [Chapter 2](ch02-python-foundations-and-environment.md): an activated `code/.venv` environment with NumPy and PyTorch installed. Check with `python -m scripts.ch02_check_env`.
- Python lists and slicing (section 2.4).

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Array / tensor | A grid of numbers, all of one type, stored together in memory. NumPy calls it an array; PyTorch calls it a tensor | 3.2 |
| Element | One number inside a tensor | 3.2 |
| Shape | The size of the grid along each direction, such as `(3, 5)` | 3.3 |
| Dimension (rank) | How many directions the grid has: the length of its shape | 3.3 |
| Axis | One of those directions, numbered from 0 | 3.3 |
| Scalar, vector, matrix | Tensors with 0, 1, and 2 dimensions | 3.3 |
| Data type (dtype) | The kind of number every element holds, such as 32-bit decimal or 64-bit integer | 3.4 |
| Precision, range | How many digits a type keeps / how large a value it can hold | 3.4 |
| View | A tensor that shares memory with another tensor, showing it differently | 3.5 |
| Contiguous | Elements laid out in memory in the order the shape implies | 3.6 |
| Broadcasting | Automatically repeating a smaller tensor to match a larger one in an operation | 3.7 |
| Reduction | An operation that combines many elements into fewer, such as sum or max | 3.8 |
| Batch, batch dimension | Several examples processed together; the axis that indexes them | 3.9 |
| Device | Where a tensor's memory lives and where operations on it run: CPU or a GPU | 3.11 |
| GPU, RAM, VRAM | Graphics processor; main memory; memory on the graphics card | 3.11 |

---

### 3.1 The problem: a model processes thousands of numbers at once

A modest language model, much smaller than any commercial one, might represent each token as a list of 768 numbers. A batch of 32 sequences of 256 tokens, with 768 numbers for every token, holds millions of numbers. The script in section 3.12 reports that this one batch occupies 24 MiB in the common 32-bit format. Every layer of the model transforms all of those numbers, many times per second during training.

Three engineering problems follow:

1. **Speed.** Processing millions of numbers one at a time with Python loops is far too slow.
2. **Structure.** The numbers have meaning only through their arrangement: which sequence, which position, which feature. Code must keep track of that arrangement through every transformation.
3. **Memory and hardware.** Those numbers must fit somewhere, in main memory or on a GPU, and the choice of number format changes how much space they take and how accurate they are.

Tensors solve all three, and they introduce new failure modes of their own. Both halves are this chapter's subject.

---

### 3.2 From Python lists to arrays: why arrays exist

A Python list of a million numbers is a million separate Python objects. When you loop over it, the interpreter examines each object, checks its type, finds the right way to multiply it, and creates a new object for the result, a million times over. An **array** stores a million raw numbers side by side in one block of memory, all of the same type. One call runs a compiled loop, written in a fast language such as C, over the whole block.

File: [`code/examples/ch03/why_arrays.py`](../../code/examples/ch03/why_arrays.py)

```python
@@FILE code/examples/ch03/why_arrays.py@@
```

Observed output (timings vary with the machine and from run to run):

```text
@@RUN python examples/ch03/why_arrays.py@@
```

The results are identical. The array operation is many times faster, and on a GPU the gap for large operations grows much larger. This is why all numerical Python code, including all of PyTorch, is written in terms of whole-array operations rather than loops over **elements** (the individual numbers inside an array).

**NumPy and PyTorch.** NumPy is Python's standard array library, used throughout science and data work. PyTorch provides its own array type, called a **tensor**, with two capabilities NumPy lacks and deep learning needs: tensors can live on a GPU, and PyTorch can automatically work out how to adjust the numbers that produced a result (Chapter 6). The two libraries are deliberately similar; most operations have the same names. This book uses PyTorch tensors for model code and NumPy for some data handling. The word *tensor* in machine learning means "multi-dimensional array"; it does not carry the specialized meaning it has in physics.

---

### 3.3 Shape, dimensions, and axes

A tensor's **shape** lists its size along each direction. Its number of **dimensions** is how many directions there are. Each direction is an **axis**, numbered from 0.

File: [`code/examples/ch03/shapes_axes.py`](../../code/examples/ch03/shapes_axes.py)

```python
@@FILE code/examples/ch03/shapes_axes.py@@
```

Observed output:

```text
@@RUN python examples/ch03/shapes_axes.py@@
```

The names in the output have conventional meanings:

| Dimensions | Name | Shape example | In this book |
|---|---|---|---|
| 0 | Scalar | `()` | A single loss value |
| 1 | Vector | `(5,)` | One sequence of token IDs |
| 2 | Matrix | `(3, 5)` | A batch of 3 sequences, 5 tokens each |
| 3 | (3-dimensional tensor) | `(3, 5, 4)` | A batch of sequences, with a list of numbers for every token |

**The meaning of an axis is a convention you must track, not something the tensor knows.** The tensor `features` has shape `(3, 5, 4)`. *We* decided that axis 0 is the batch, axis 1 the position in the sequence, and axis 2 the features. PyTorch has no idea; it would happily sum over the wrong axis. That is why model code in this book writes the intended layout in comments next to every important tensor, like `# (batch, sequence, features)`, and why shape traces (Chapter 16.7) are such an effective debugging tool. Appendix C (planned) collects the layouts used in the book.

**Operations along an axis.** "Sum along axis 0" combines elements that differ only in their axis-0 position: for the `(3, 5)` matrix, it adds the three sequences position by position and returns 5 results. "Sum along axis 1" adds across each row and returns one result per sequence. A helpful way to remember: the axis you name is the one that *disappears*. Axis `-1` means the last axis, whatever the number of dimensions, and model code uses it constantly for "the features of each token".

---

### 3.4 Data types: integers, float32, float16, bfloat16

Every element of a tensor has the same **data type** (dtype). The dtype decides what values can be stored and how many bytes each takes.

File: [`code/examples/ch03/dtypes.py`](../../code/examples/ch03/dtypes.py)

```python
@@FILE code/examples/ch03/dtypes.py@@
```

Observed output:

```text
@@RUN python examples/ch03/dtypes.py@@
```

What the output shows:

- **Defaults.** Whole numbers become `int64` (8-byte integers), decimals become `float32` (4-byte floating-point numbers). Token IDs are `int64` in PyTorch because the functions that look up tables by ID expect that type.
- **Precision.** A floating-point number stores a limited number of significant digits. float32 keeps about 7 decimal digits, so `0.6` is stored as the nearest value it can represent, which prints as `0.6000000238418579` when shown in full. The 2-byte formats keep far fewer digits: float16 stored one third as `0.33325...`, and bfloat16 as `0.33398...`.
- **Range.** float16 cannot hold values above about 65,000, so 70,000 became infinity (`inf`). bfloat16 trades precision for range: it held 70,000 approximately (as 70,144) but kept fewer digits than float16. That trade is why bfloat16 is popular for training: values rarely overflow to infinity, and the loss of digits is usually tolerable (Chapter 19.10).
- **Silent overflow in small integers.** An `int8` holds values from -128 to 127. Adding 10 to 120 wrapped around to -126 with no error. Integer overflow is rare in this book, but this behavior shows why the type of a tensor deserves a look whenever results seem strange.
- **Mixing types.** Some operations refuse to mix integers and decimals (the matrix multiplication raised an error), while others quietly convert. Converting with `.to(...)` is explicit; converting decimals to integers drops the fractional part (`2.7` and `-2.7` became `2` and `-2`).

The common types, and where the book uses them:

| dtype | Bytes per element | Use in this book |
|---|---|---|
| `int64` | 8 | Token IDs, labels |
| `bool` | 1 | Masks (which positions to use or ignore) |
| `float32` | 4 | Default for parameters and computation; the safe choice |
| `bfloat16` | 2 | Faster, smaller training and inference on hardware that supports it (Ch 19.10, 24) |
| `float16` | 2 | Same idea as bfloat16, more precision, much less range; needs care to avoid overflow |
| 8-bit and 4-bit formats | 1 or less | Quantized models (Ch 24.4) |

---

### 3.5 Indexing and slicing

Indexing a tensor works like indexing a list, extended to several axes: you give one index or slice per axis, separated by commas.

File: [`code/examples/ch03/indexing.py`](../../code/examples/ch03/indexing.py)

```python
@@FILE code/examples/ch03/indexing.py@@
```

Observed output:

```text
@@RUN python examples/ch03/indexing.py@@
```

The patterns worth memorizing, because you will use each one in later chapters:

- `batch[:, 0]` and `batch[:, -1]`: one position from every sequence. A colon means "everything along this axis". Generation (Chapter 17) takes the model's scores at the last position of every sequence with exactly this pattern.
- `batch[:, :-1]` and `batch[:, 1:]`: every sequence without its last token, and without its first. These are the inputs and targets of next-token training (Chapter 11), the shift you met in section 2.5.
- `table[ids]`: indexing with a tensor of integers picks those rows, in that order, repeats allowed. Shape `(3,)` of IDs gave shape `(3, 2)`: one row of the table per ID. This is precisely what an embedding table does (Chapter 10).
- `batch[mask]`: a boolean tensor of the same shape selects the elements where it is `True`. The result is flat, because the selected elements no longer form a grid.

**Views share memory.** A basic slice such as `batch[0]` does not copy anything: it is a **view**, a new tensor object that looks at the same block of memory. Writing through the view changed the original. This is efficient, and occasionally a source of bugs when code modifies what it thinks is a private copy. When you need an independent copy, call `.clone()`. (Indexing with a tensor of IDs or a boolean mask, by contrast, makes a copy.)

---

### 3.6 Reshaping, views, transposes, and contiguity

The same numbers can be arranged in different shapes. Model code rearranges tensors constantly, for example to split the feature axis into several groups for multi-head attention (Chapter 14).

File: [`code/examples/ch03/reshaping.py`](../../code/examples/ch03/reshaping.py)

```python
@@FILE code/examples/ch03/reshaping.py@@
```

Observed output:

```text
@@RUN python examples/ch03/reshaping.py@@
```

- **`reshape`** arranges the same elements, in the same order, into a new shape. The total count must match: 12 elements fit `(3, 4)` or `(2, 6)` but not `(5, anything)`. Passing `-1` for one axis asks PyTorch to work out that size.
- **`unsqueeze` and `squeeze`** add or remove an axis of size 1. They change how the tensor *lines up* with others, not what it contains. `seq.unsqueeze(0)` turns one sequence into a batch containing one sequence, which a model expecting `(batch, sequence)` input needs.
- **`transpose(a, b)`** swaps two axes. It *changes the order in which elements are visited*: compare `m.reshape(3, 2)`, which keeps the reading order 0, 1, 2, 3, 4, 5, with `m.transpose(0, 1)`, which turns rows into columns. Confusing the two is a classic bug that produces a tensor of the right shape holding scrambled data. No error is raised.
- **Contiguity.** A transpose does not move any data; it changes how PyTorch steps through memory. Afterwards the elements are no longer laid out in the order the new shape implies, and the tensor is called non-**contiguous**. `view` (a reshape that guarantees no copying) refuses to work on such tensors. `reshape` copies when it has to, and `.contiguous()` makes a laid-out-in-order copy explicitly. In practice: use `reshape`, unless you have a specific reason to require a no-copy view.

---

### 3.7 Broadcasting: combining tensors of different shapes

Operations such as `+` and `*` work element by element on two tensors of the same shape. **Broadcasting** extends them to tensors of different shapes, by repeating the smaller one, without actually copying it in memory, until the shapes match.

File: [`code/examples/ch03/broadcasting.py`](../../code/examples/ch03/broadcasting.py)

```python
@@FILE code/examples/ch03/broadcasting.py@@
```

Observed output:

```text
@@RUN python examples/ch03/broadcasting.py@@
```

The rule, in words: line the two shapes up **from the right**. Compare them axis by axis. Two sizes are compatible if they are equal or if one of them is 1; a size of 1 is repeated to match. If one tensor has fewer axes, it is treated as having extra size-1 axes on the left. If any pair of sizes is incompatible, PyTorch raises an error.

| Shapes | Lined up from the right | Result |
|---|---|---|
| `(2, 3)` and `()` | a single number matches everything | `(2, 3)` |
| `(2, 3)` and `(3,)` | 3 matches 3; the missing axis is repeated | `(2, 3)` |
| `(2, 3)` and `(2, 1)` | 3 vs 1: repeat; 2 matches 2 | `(2, 3)` |
| `(2, 3)` and `(2,)` | 3 vs 2: incompatible | error |
| `(3,)` and `(3, 1)` | 3 vs 1: repeat; nothing vs 3: repeat | `(3, 3)`, **the silent bug** |

**The silent bug.** `predictions` had shape `(3,)` and `targets` had shape `(3, 1)`, as can happen after an accidental `unsqueeze`. Instead of three differences, broadcasting produced a 3 by 3 grid comparing every prediction with every target, and the average difference came out as 0.89 instead of 0. No error, no warning, a plausible-looking number. In a training loop, this kind of bug can make the loss decrease while the model learns the wrong thing. The defense is to check shapes at every step where two tensors meet, and to write tests that assert them (Chapter 16.8).

Broadcasting is genuinely useful. Model code adds a learned list of numbers to every position of every sequence with one `+`, relying on exactly the second row of the table above.

---

### 3.8 Operations along an axis: sum, mean, max, argmax

A **reduction** combines many elements into fewer. Model code reduces along a chosen axis.

File: [`code/examples/ch03/reductions.py`](../../code/examples/ch03/reductions.py)

```python
@@FILE code/examples/ch03/reductions.py@@
```

Observed output:

```text
@@RUN python examples/ch03/reductions.py@@
```

- `argmax(dim=1)` returns the *position* of the largest score in each row, not the score itself. For a row of scores over candidate tokens, that position is the token ID of the greedy choice (Chapter 1.7). Generation code in Chapter 17 does exactly this.
- `max(dim=...)` returns both the values and their positions (`.values` and `.indices`). `topk(k, dim=...)` returns the k largest, which is the core of top-k decoding (Chapter 21.5).
- **`keepdim=True`** keeps the reduced axis with size 1 instead of removing it. Removing it would leave `(2,)`, which broadcasts against `(2, 5)` incorrectly or not at all. Keeping it leaves `(2, 1)`, which broadcasts row by row, so "subtract each row's maximum from that row" works. You will see this precise pattern inside implementations of softmax (Chapter 5.7).

---

### 3.9 Batching: adding a batch dimension

Models process a **batch** of examples in one call, for speed (one large operation instead of many small ones) and, during training, for more stable learning (Chapter 6). By convention the batch is axis 0, the **batch dimension**.

File: [`code/examples/ch03/batching.py`](../../code/examples/ch03/batching.py)

```python
@@FILE code/examples/ch03/batching.py@@
```

Observed output:

```text
@@RUN python examples/ch03/batching.py@@
```

- `torch.stack` adds a new axis and places the tensors along it: three `(4,)` sequences become one `(3, 4)` batch. `torch.cat` joins tensors along an *existing* axis: three `(4,)` become one `(12,)`. Using `cat` where you meant `stack` produces one long sequence instead of a batch.
- `stack` needs equal shapes. Real sentences have different lengths, so they must be padded to a common length first. Exercise 5 builds that, and Chapter 11 develops it fully.
- Batched computation gives the same results as processing each example separately; the batch is a container, not a mixing step. (Chapter 13 introduces masks precisely to keep padded positions from mixing in.)
- A single example given to a model that expects a batch needs `unsqueeze(0)`. Forgetting it is one of the most common shape errors.

---

### 3.10 Moving between NumPy and PyTorch

File: [`code/examples/ch03/numpy_torch.py`](../../code/examples/ch03/numpy_torch.py)

```python
@@FILE code/examples/ch03/numpy_torch.py@@
```

Observed output:

```text
@@RUN python examples/ch03/numpy_torch.py@@
```

Two traps:

- `torch.from_numpy` shares memory with the NumPy array, so changes to one show up in the other. `torch.tensor(array)` copies.
- **NumPy's default decimal type is float64; PyTorch's is float32.** A tensor made from a NumPy array keeps float64. Feeding it to a float32 model produces a type error, or doubles memory use. Convert explicitly with `.to(torch.float32)`.

Use `.tolist()` to get plain Python values, for printing or for saving to JSON.

---

### 3.11 Hardware: CPU, GPU, RAM, VRAM, and devices in PyTorch

#### What each piece of hardware does

- The **CPU** (central processing unit) runs your program. It has a handful of powerful cores that are good at complicated, branching work. The test machine for this book has 20 logical cores, of which PyTorch uses 14 by default.
- **RAM** is the computer's main memory, where your program's data normally lives. The test machine has 14 GiB.
- A **GPU** (graphics processing unit) has thousands of simpler cores that apply the same operation to many numbers at once. That is exactly the shape of neural-network work, so large tensor operations run many times faster on a GPU.
- **VRAM** is the GPU's own memory. A tensor must be in VRAM for the GPU to operate on it, and VRAM is usually much smaller than RAM. On consumer graphics cards it is often between 8 and 24 GB. **VRAM is the most common hard limit in deep learning**: if a model, its training state, and a batch of data do not fit in VRAM together, training will not run on that GPU.
- Apple Silicon Macs have a GPU that shares the system's main memory; PyTorch calls it **MPS**.

#### Devices in PyTorch

Every tensor lives on a **device**: `cpu`, `cuda` (an NVIDIA GPU, named after NVIDIA's CUDA software), or `mps`. Operations run on the device where their inputs live, and all inputs to an operation must be on the same device. `tensor.to(device)` copies a tensor to another device. Copying is not free: it moves data across a connection that is much slower than either device's own memory, so well-written code moves data to the GPU once and keeps it there.

The book's helper module chooses a device and measures memory:

File: [`code/llmfp/devices.py`](../../code/llmfp/devices.py)

```python
@@FILE code/llmfp/devices.py@@
```

Every PyTorch script from here on accepts `--device auto|cpu|cuda|mps` via `add_device_argument`, and calls `pick_device`. With `auto`, an NVIDIA GPU is preferred, then an Apple GPU, then the CPU. Asking for a device that is not available is an immediate error with a suggestion, rather than a confusing failure later.

---

### 3.12 Measuring how much memory a tensor uses

A tensor's memory is its number of elements times the bytes per element for its dtype; `tensor_bytes` computes exactly that. This lets you estimate, *before* running anything, whether a model or a batch will fit.

File: [`code/scripts/ch03_tensor_tour.py`](../../code/scripts/ch03_tensor_tour.py)

```python
@@FILE code/scripts/ch03_tensor_tour.py@@
```

Observed output on the test machine (CPU only):

```text
@@RUN python -m scripts.ch03_tensor_tour@@
```

And asking for a GPU that is not there:

```text
@@RUN python -m scripts.ch03_tensor_tour --device cuda || true@@
```

Reading the output:

- Halving the bytes per element halves the memory, which is one reason 2-byte formats matter for large models.
- Holding the parameters of a 124-million-parameter model (about the size of the smallest GPT-2) in float32 takes about 473 MiB. Training needs several times more than the parameters alone, for reasons Chapter 19 explains, and those extra amounts are what usually exhaust a GPU's VRAM.
- The script times a *matrix multiplication* (`a @ b`), the operation at the heart of neural-network layers, which Chapter 5 explains by its behavior. Here it serves only as a benchmark. On a machine with a GPU, the script also times the copy to the device and the same operation there. **The author did not run the GPU branch**, because the test machine has no GPU.

> **Memory measured here is a lower bound.** `tensor_bytes` counts the numbers a tensor holds. A running program also uses memory for intermediate results, library overhead, and (on GPUs) memory reserved but not yet used. Chapter 19 and Chapter 39 measure real memory use of training and inference.

---

### 3.13 Habits for debugging shapes

Shape errors come in two kinds: loud (PyTorch raises an error) and silent (broadcasting or a wrong reshape produces a tensor of a plausible shape holding the wrong data). These habits catch both:

1. **Write the layout down.** Next to every important tensor, comment its intended axes: `# (batch, sequence, features)`.
2. **Print shapes, not values.** When something fails, print `tensor.shape`, `tensor.dtype`, and `tensor.device` for every input to the failing line. Values are rarely informative at first; shapes usually are.
3. **Read error messages for the numbers.** `The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1` tells you which axis disagreed and what the two sizes were.
4. **Assert shapes in code.** `assert logits.shape == (batch_size, seq_len, vocab_size), logits.shape` turns a silent bug into a loud one at the point where it happens. Model code in Part 3 does this at key points.
5. **Use small, distinct sizes in tests.** If batch, sequence, and feature sizes are all 4, a transposed tensor has the "right" shape. Use sizes like 2, 3, and 5 so every axis is distinguishable.
6. **Use `arange` values in experiments.** `torch.arange(12).reshape(3, 4)` makes it obvious where each element ended up after a reshape or transpose.
7. **Watch for unexpected growth.** A result with more dimensions or more elements than you expected, like `(3, 3)` instead of `(3,)`, is the signature of accidental broadcasting.

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension N` | Shapes cannot broadcast | Print both shapes; fix the axis named in the message |
| A result has an unexpected extra axis, or a loss looks plausible but wrong | Silent broadcasting, such as `(N,)` against `(N, 1)` | Assert shapes; `squeeze` or `unsqueeze` deliberately |
| `expected m1 and m2 to have the same dtype` / `expected scalar type Float but found Long` | Integers mixed with decimals | `.to(torch.float32)` for computed values; keep IDs as int64 |
| Model input has wrong number of dimensions | Single example without a batch axis | `x.unsqueeze(0)` |
| `view size is not compatible with input tensor's size and stride` | `view` after a transpose | Use `reshape`, or `.contiguous().view(...)` |
| `Expected all tensors to be on the same device` | One tensor on CPU, one on GPU | `.to(device)` for every input; create new tensors with `device=` |
| Changing a slice changed the original | Slices are views | `.clone()` when you need a private copy |
| A model is unexpectedly slow or uses double memory | float64 tensors from NumPy | Convert to float32 |
| `CUDA out of memory` | Tensors or model too large for VRAM | Smaller batch, shorter sequences, 2-byte dtypes (Ch 19.10); estimate first with `tensor_bytes` |

---

### 3.14 Recap, concept checks, exercises, answers, and checkpoint

#### Recap

- A tensor is a grid of numbers of one dtype, stored together in memory, which makes whole-array operations far faster than Python loops.
- **Shape** gives the size along each **axis**; what each axis means is a convention you must track and document.
- **Dtype** trades memory for precision and range: float32 is the safe default, bfloat16 keeps range with fewer digits, float16 keeps digits with little range, int64 holds token IDs, bool holds masks.
- Slices are **views** that share memory; ID and mask indexing copy. `table[ids]` is a table lookup.
- `reshape` keeps element order; `transpose` changes it. Use `reshape` over `view` unless you need a guaranteed view.
- **Broadcasting** lines shapes up from the right and repeats size-1 axes. It is powerful, and it can silently produce the wrong shape.
- Reductions remove the named axis unless `keepdim=True`. `argmax` finds the position of the largest score.
- Batches stack examples along axis 0; equal shapes are required, hence padding.
- Tensors live on a **device**; operations need all inputs on one device. VRAM is usually the binding limit. Memory equals element count times bytes per element, as a lower bound.

#### Concept checks

1. Why is `array_a * array_b` faster than a list comprehension over the same numbers?
2. A tensor has shape `(8, 128, 64)`. How many dimensions does it have? If it is `(batch, sequence, features)`, what does `t[2, 5]` hold, and what is its shape?
3. What does `t.sum(dim=-1)` do to a `(8, 128, 64)` tensor? What shape results?
4. Why might a training setup choose bfloat16 over float16?
5. You run `row = batch[0]; row[:] = 0`. What happens to `batch`? How would you avoid that?
6. What is the difference between `x.reshape(4, 3)` and `x.transpose(0, 1)` for an `x` of shape `(3, 4)`?
7. What shape does `torch.ones(5) + torch.ones(5, 1)` produce, and why is that dangerous?
8. Why does `scores - scores.max(dim=1, keepdim=True).values` work, when the same expression without `keepdim=True` would not do what you meant?
9. What is the difference between `torch.stack` and `torch.cat`?
10. You create a tensor with `torch.from_numpy(np.array([0.5, 1.5]))`. What dtype does it have, and why might that cause a problem?
11. Why should a training script move its model to the GPU once rather than on every step?
12. Roughly how much memory do a model's parameters need in float32 versus bfloat16, if the model has one billion parameters? Explain how you would work it out.

#### Exercises

**Exercise 1 (predict shapes).** Without running code, write down the shapes of: `torch.zeros(4, 6)[:, 2:]`, `torch.zeros(4, 6)[1]`, `torch.zeros(4, 6).reshape(2, -1, 3)`, `torch.zeros(4, 6).unsqueeze(1)`, `torch.zeros(4, 6).sum(dim=0)`, `torch.zeros(4, 6).max(dim=1, keepdim=True).values`, `torch.zeros(4, 1) + torch.zeros(6)`. Then check each one in Python.

**Exercise 2 (the transpose trap).** Create `x = torch.arange(6).reshape(2, 3)`. Produce a `(3, 2)` tensor with `reshape` and another with `transpose`. Write a one-line test that would fail if code used one where it should have used the other.

**Exercise 3 (find the broadcasting bug).** This function should return the average absolute difference between predictions and targets, both lists of the same length:

```python
def mean_abs_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return (predictions - targets).abs().mean().item()
```

It gives a wrong answer when called as `mean_abs_error(torch.tensor([1.0, 2.0]), torch.tensor([[1.0], [2.0]]))`. Fix it so it raises a clear error for mismatched shapes instead, and write a pytest test for both the correct case and the error.

**Exercise 4 (memory estimates).** Using `tensor_bytes` and `format_bytes`, write a short script that prints the memory needed for a batch of token *IDs* (int64) and a batch of token *features* (float32 and bfloat16) for batch sizes 8, 32, and 128, with sequence length 512 and 768 features. Which grows faster, and why does that matter for choosing a batch size?

**Exercise 5 (pad and stack).** Write `pad_and_stack(sequences, pad_id=0)` that takes a list of token-ID lists of different lengths and returns two tensors of shape `(batch, longest length)`: the padded IDs (int64), and a boolean mask that is `True` at real tokens. Write tests, including the case where `pad_id` is also a real token ID.

**Exercise 6 (devices, if you have a GPU).** If you have an NVIDIA or Apple GPU, run `python -m scripts.ch03_tensor_tour --size 4096` and compare CPU and GPU times, including the copy time. At what `--size` does the GPU stop being worth the copy for a single multiplication? If you have no GPU, explain from the script's code what it would measure, and run it in a free cloud notebook if you have access to one.

#### Suggested answers and acceptance criteria

**Concept checks**

1. The array stores raw numbers of one type contiguously, and one call runs a compiled loop over all of them. The list comprehension makes the interpreter handle each Python object individually.
2. Three. `t[2, 5]` is the features of token 5 in sequence 2: shape `(64,)`.
3. It adds up the 64 features of each token, producing `(8, 128)`: the last axis disappears.
4. bfloat16 has a much larger range, so large values do not overflow to infinity. It keeps fewer digits, which training usually tolerates.
5. `batch[0]` becomes all zeros, because `row` is a view of the same memory. Use `row = batch[0].clone()`.
6. `reshape` keeps the reading order of elements (0, 1, 2, ... filled row by row into the new shape); `transpose` turns rows into columns, changing which element sits where. Both give shape `(4, 3)`; the contents differ.
7. `(5, 5)`. The `(5,)` and `(5, 1)` shapes broadcast against each other into a grid instead of pairing elements, with no error, so code computing "differences" or "losses" silently gets a wrong but plausible number.
8. With `keepdim=True` the maximum has shape `(rows, 1)`, which broadcasts across each row. Without it, the shape is `(rows,)`, which lines up with the *columns* instead (or fails), subtracting the wrong values.
9. `stack` creates a new axis (three `(4,)` become `(3, 4)`); `cat` joins along an existing axis (three `(4,)` become `(12,)`).
10. float64, NumPy's default. A float32 model will reject it with a type error or use twice the memory and run slower.
11. Copying between devices is slow compared with computation on the device; copying the model every step would waste most of the time.
12. One billion parameters at 4 bytes each is about 4 GB (about 3.7 GiB); at 2 bytes, about 2 GB. Multiply the parameter count by the bytes per element of the dtype, which `tensor_bytes` does for real tensors.

**Exercise 1.** `(4, 4)`, `(6,)`, `(2, 4, 3)`, `(4, 1, 6)`, `(6,)`, `(4, 1)`, `(4, 6)`. Acceptance: you check each in Python and explain any you got wrong. The last one is broadcasting: `(4, 1)` and `(6,)` become `(4, 6)`.

**Exercise 2.** `x.reshape(3, 2)` is `[[0, 1], [2, 3], [4, 5]]`; `x.transpose(0, 1)` is `[[0, 3], [1, 4], [2, 5]]`. A test such as `assert x.transpose(0, 1)[0].tolist() == [0, 3]` fails if `reshape` is used. Acceptance: your test distinguishes the two, and you can explain why using `arange` values (rather than zeros or ones) is what makes the difference visible.

**Exercise 3.** One fix:

```python
def mean_abs_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    if predictions.shape != targets.shape:
        raise ValueError(f"shape mismatch: predictions {tuple(predictions.shape)} vs targets {tuple(targets.shape)}")
    return (predictions - targets).abs().mean().item()
```

Acceptance: `mean_abs_error(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))` returns `0.0`; the mismatched call raises `ValueError` naming both shapes; your test uses `pytest.raises`. Note the design choice: the function refuses to guess, instead of quietly squeezing, because a caller passing the wrong shape probably has a bug elsewhere.

**Exercise 4.** Acceptance: your table shows IDs costing 8 bytes per token and features costing 768 numbers per token (3 KiB in float32), so the features dominate by far, and both grow in proportion to batch size. This matters because the activations (Chapter 5) of every layer have the features' shape, so batch size is one of the main levers for fitting training into memory (Chapter 19.8).

**Exercise 5.** Solution: [`code/solutions/ch03_pad_and_stack.py`](../../code/solutions/ch03_pad_and_stack.py), tests in [`code/tests/test_ch03_solutions.py`](../../code/tests/test_ch03_solutions.py). Observed output:

```text
@@RUN python -m solutions.ch03_pad_and_stack@@
```

Acceptance: shapes `(batch, longest)`; dtypes int64 and bool; a test showing that when `pad_id` is a real token ID, only the mask can tell padding from content, which is exactly why models receive a mask and do not rely on the pad ID (Chapter 11.6).

**Exercise 6.** Acceptance (GPU owners): a small table of CPU time, copy time, and GPU time for at least three sizes, and a sentence on where the copy cost outweighs the speed-up. Expect the GPU to win clearly for large sizes and lose for small ones, because the fixed costs of starting work on the GPU and copying data dominate small operations. Acceptance (without a GPU): an explanation that the script times the CPU multiplication, then the copy to the device, then the device multiplication, and that `.item()` forces the program to wait until the device has actually finished, because GPU operations otherwise return before the work is done and timings would be meaningless.

#### Checkpoint: what you can now do independently

You can now:

- Create, inspect, and rearrange tensors, predicting their shapes before running code.
- Choose dtypes deliberately and explain the precision, range, and memory trade-offs.
- Recognize views versus copies, reshape versus transpose, and the signature of an accidental broadcast.
- Select a device from the command line and estimate the memory a tensor or a model's parameters will need.
- Debug a shape error by printing shapes, dtypes, and devices, and guard against silent errors with assertions and well-chosen test sizes.

**Next:** Chapter 4 makes experiments trustworthy: splitting data so evaluation is honest, recognizing leakage and overfitting, and recording every run so it can be reproduced.
