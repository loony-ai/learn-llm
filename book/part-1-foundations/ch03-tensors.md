## Chapter 3: Tensors: NumPy and PyTorch as Containers of Numbers

[Back to index](../../README.md) · Previous: [Chapter 2](ch02-python-foundations-and-environment.md) · Next: [Chapter 4](ch04-data-experiments-reproducibility.md)

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
"""Chapter 3.2: why arrays exist. Compare a Python loop with a NumPy array operation.

Run from `code/`:  python examples/ch03/why_arrays.py
Timings depend on your machine; the ratio is what matters.
"""

import time

import numpy as np

size = 1_000_000
python_a = [float(i) for i in range(size)]
python_b = [2.0] * size
array_a = np.arange(size, dtype=np.float64)
array_b = np.full(size, 2.0)

start = time.perf_counter()
python_result = [x * y for x, y in zip(python_a, python_b)]
python_seconds = time.perf_counter() - start

start = time.perf_counter()
array_result = array_a * array_b  # one operation applied to every element
array_seconds = time.perf_counter() - start

print(f"Python list loop : {python_seconds * 1000:8.2f} ms")
print(f"NumPy array op   : {array_seconds * 1000:8.2f} ms")
print(f"Speed-up         : about {python_seconds / array_seconds:.0f}x")
print("Same results     :", python_result[:3] == array_result[:3].tolist(), python_result[:3])

# Why: a list holds a million separate Python objects, each examined one at a time
# by the interpreter. An array holds a million raw numbers packed side by side in
# memory, all of one type, and one call runs a compiled loop over all of them.
print("List element type :", type(python_a[0]).__name__, "| array element type:", array_a.dtype)
print("Array memory      :", array_a.nbytes, "bytes =", array_a.nbytes // size, "bytes per number")
```

Observed output (timings vary with the machine and from run to run):

```text
Python list loop :    49.16 ms
NumPy array op   :    11.21 ms
Speed-up         : about 4x
Same results     : True [0.0, 2.0, 4.0]
List element type : float | array element type: float64
Array memory      : 8000000 bytes = 8 bytes per number
```

The results are identical. The array operation is many times faster, and on a GPU the gap for large operations grows much larger. This is why all numerical Python code, including all of PyTorch, is written in terms of whole-array operations rather than loops over **elements** (the individual numbers inside an array).

**NumPy and PyTorch.** NumPy is Python's standard array library, used throughout science and data work. PyTorch provides its own array type, called a **tensor**, with two capabilities NumPy lacks and deep learning needs: tensors can live on a GPU, and PyTorch can automatically work out how to adjust the numbers that produced a result (Chapter 6). The two libraries are deliberately similar; most operations have the same names. This book uses PyTorch tensors for model code and NumPy for some data handling. The word *tensor* in machine learning means "multi-dimensional array"; it does not carry the specialized meaning it has in physics.

---

### 3.3 Shape, dimensions, and axes

A tensor's **shape** lists its size along each direction. Its number of **dimensions** is how many directions there are. Each direction is an **axis**, numbered from 0.

File: [`code/examples/ch03/shapes_axes.py`](../../code/examples/ch03/shapes_axes.py)

```python
"""Chapter 3.3: shape, dimensions, and axes, using token IDs and features.

Run from `code/`:  python examples/ch03/shapes_axes.py
"""

import torch

scalar = torch.tensor(7)                         # one number
vector = torch.tensor([5, 9, 2, 5, 7])           # one sequence of token IDs
matrix = torch.tensor([[5, 9, 2, 5, 7],          # a batch: 3 sequences of 5 token IDs
                       [5, 1, 4, 0, 3],
                       [8, 9, 2, 6, 7]])
for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    print(f"{name:<7} shape={tuple(t.shape)!s:<8} dimensions={t.ndim}  elements={t.numel()}")

# A 3-dimensional tensor: for every token in every sequence, 4 numbers describing it.
# This (batch, sequence, features) layout is the one a transformer works with.
features = torch.arange(3 * 5 * 4).reshape(3, 5, 4)
print("\nfeatures shape:", tuple(features.shape), "-> (batch=3, sequence=5, features=4)")
print("features[0] is sequence 0, shape", tuple(features[0].shape))
print("features[0, 1] is token 1 of sequence 0, shape", tuple(features[0, 1].shape), "->", features[0, 1].tolist())

# An axis is one dimension, numbered from 0. Operations "along an axis" combine
# the elements that differ only in that axis.
print("\nmatrix:\n", matrix)
print("sum along axis 0 (down the columns, one result per position):", matrix.sum(dim=0).tolist())
print("sum along axis 1 (across each row, one result per sequence):  ", matrix.sum(dim=1).tolist())
print("negative axis -1 means the last axis:", matrix.sum(dim=-1).tolist())
```

Observed output:

```text
scalar  shape=()       dimensions=0  elements=1
vector  shape=(5,)     dimensions=1  elements=5
matrix  shape=(3, 5)   dimensions=2  elements=15

features shape: (3, 5, 4) -> (batch=3, sequence=5, features=4)
features[0] is sequence 0, shape (5, 4)
features[0, 1] is token 1 of sequence 0, shape (4,) -> [4, 5, 6, 7]

matrix:
 tensor([[5, 9, 2, 5, 7],
        [5, 1, 4, 0, 3],
        [8, 9, 2, 6, 7]])
sum along axis 0 (down the columns, one result per position): [18, 19, 8, 11, 17]
sum along axis 1 (across each row, one result per sequence):   [28, 13, 32]
negative axis -1 means the last axis: [28, 13, 32]
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
"""Chapter 3.4: data types and why they matter.

Run from `code/`:  python examples/ch03/dtypes.py
"""

import torch

token_ids = torch.tensor([5, 9, 2])
values = torch.tensor([0.5, 1.25])
print("integers default to", token_ids.dtype, "| decimals default to", values.dtype)

# Storing the same value with fewer bytes keeps fewer digits of precision.
third = 1 / 3
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    t = torch.tensor([third], dtype=dtype)
    print(f"{str(dtype):<15} bytes/element={t.element_size()}  stores 1/3 as {t.item():.10f}")

# Most decimals cannot be stored exactly in binary, so you see the nearest stored value.
print("0.6 in float32   :", torch.tensor(0.6).item(), "(nearest float32 value, shown in full)")

# float16 has a small range: large values become infinity.
print("70000 in float16 :", torch.tensor([70000.0], dtype=torch.float16).item())
print("70000 in bfloat16:", torch.tensor([70000.0], dtype=torch.bfloat16).item())

# Small integer types wrap around silently when they overflow.
small = torch.tensor([120], dtype=torch.int8)
print("int8 120 + 10    :", (small + 10).item(), "(wrapped around; no error)")

# Some operations refuse to mix types; others convert silently.
weights = torch.ones(3, 2)  # float32
try:
    token_ids @ weights  # matrix multiplication of int64 with float32
except RuntimeError as error:
    print("int64 @ float32  : RuntimeError -", str(error).splitlines()[0])
print("int64 + float32  :", (token_ids + torch.tensor([0.5, 0.5, 0.5])).dtype, "(converted automatically)")
print("converted        :", token_ids.to(torch.float32), "| back:", torch.tensor([2.7, -2.7]).to(torch.int64))
```

Observed output:

```text
integers default to torch.int64 | decimals default to torch.float32
torch.float32   bytes/element=4  stores 1/3 as 0.3333333433
torch.float16   bytes/element=2  stores 1/3 as 0.3332519531
torch.bfloat16  bytes/element=2  stores 1/3 as 0.3339843750
0.6 in float32   : 0.6000000238418579 (nearest float32 value, shown in full)
70000 in float16 : inf
70000 in bfloat16: 70144.0
int8 120 + 10    : -126 (wrapped around; no error)
int64 @ float32  : RuntimeError - expected m1 and m2 to have the same dtype, but got: long int != float
int64 + float32  : torch.float32 (converted automatically)
converted        : tensor([5., 9., 2.]) | back: tensor([ 2, -2])
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
"""Chapter 3.5: indexing and slicing.

Run from `code/`:  python examples/ch03/indexing.py
"""

import torch

batch = torch.tensor([[10, 11, 12, 13, 14],
                      [20, 21, 22, 23, 24],
                      [30, 31, 32, 33, 34]])
print("batch[1]          ->", batch[1].tolist(), "(one row)")
print("batch[1, 3]       ->", batch[1, 3].item(), "(one element)")
print("batch[:, 0]       ->", batch[:, 0].tolist(), "(first column: position 0 of every sequence)")
print("batch[:, -1]      ->", batch[:, -1].tolist(), "(last position of every sequence)")
print("batch[:, :-1]     ->", batch[:, :-1].tolist(), "(all but the last position)")
print("batch[:, 1:]      ->", batch[:, 1:].tolist(), "(all but the first position)")
print("batch[0, ::2]     ->", batch[0, ::2].tolist(), "(every second element)")

# Indexing with a list or tensor of positions picks several rows at once.
# This is exactly how an embedding table is looked up (Chapter 10).
table = torch.tensor([[0.0, 0.0], [0.25, 0.5], [0.75, 1.0], [1.25, 1.5]])  # 4 rows of 2 numbers
ids = torch.tensor([3, 1, 3])
print("\ntable[ids]        ->", table[ids].tolist(), "shape", tuple(table[ids].shape))

# A boolean mask selects the elements where it is True.
mask = batch > 22
print("batch > 22        ->", mask[1].tolist(), "(row 1 of the mask)")
print("batch[mask]       ->", batch[mask].tolist())

# A slice is a VIEW: it shares memory with the original. Changing it changes both.
row = batch[0]
row[0] = 99
print("\nafter row[0] = 99, batch[0] ->", batch[0].tolist(), "(the original changed)")
copy = batch[1].clone()
copy[0] = -1
print("after copy[0] = -1, batch[1] ->", batch[1].tolist(), "(clone() made an independent copy)")

try:
    batch[3]
except IndexError as error:
    print("\nbatch[3] -> IndexError:", error)
```

Observed output:

```text
batch[1]          -> [20, 21, 22, 23, 24] (one row)
batch[1, 3]       -> 23 (one element)
batch[:, 0]       -> [10, 20, 30] (first column: position 0 of every sequence)
batch[:, -1]      -> [14, 24, 34] (last position of every sequence)
batch[:, :-1]     -> [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]] (all but the last position)
batch[:, 1:]      -> [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]] (all but the first position)
batch[0, ::2]     -> [10, 12, 14] (every second element)

table[ids]        -> [[1.25, 1.5], [0.25, 0.5], [1.25, 1.5]] shape (3, 2)
batch > 22        -> [False, False, False, True, True] (row 1 of the mask)
batch[mask]       -> [23, 24, 30, 31, 32, 33, 34]

after row[0] = 99, batch[0] -> [99, 11, 12, 13, 14] (the original changed)
after copy[0] = -1, batch[1] -> [20, 21, 22, 23, 24] (clone() made an independent copy)

batch[3] -> IndexError: index 3 is out of bounds for dimension 0 with size 3
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
"""Chapter 3.6: reshaping, views, transposes, and contiguity.

Run from `code/`:  python examples/ch03/reshaping.py
"""

import torch

t = torch.arange(12)
print("original      ", tuple(t.shape), t.tolist())
print("reshape(3, 4) \n", t.reshape(3, 4))
print("reshape(2, -1)", tuple(t.reshape(2, -1).shape), "(-1 means: work out this size)")
try:
    t.reshape(5, -1)
except RuntimeError as error:
    print("reshape(5, -1) RuntimeError:", error)

# Adding and removing size-1 dimensions.
seq = torch.tensor([5, 9, 2])
print("\nunsqueeze(0)  ", tuple(seq.unsqueeze(0).shape), "(a batch containing one sequence)")
print("unsqueeze(-1) ", tuple(seq.unsqueeze(-1).shape), "(each token gets its own row)")
print("squeeze()     ", tuple(seq.unsqueeze(0).squeeze().shape))

# Transposing swaps two axes. Multi-head attention (Chapter 14) splits the
# feature axis into (heads, features per head) and then moves the heads axis.
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)       # (batch, sequence, features)
split = x.reshape(2, 3, 2, 2)                      # (batch, sequence, heads, per_head)
moved = split.transpose(1, 2)                      # (batch, heads, sequence, per_head)
print("\nsplit features ", tuple(split.shape), "-> transpose(1, 2) ->", tuple(moved.shape))

# view() never copies, so it needs the elements laid out in order in memory.
# After a transpose they are not, so view() fails; reshape() copies if needed.
print("is_contiguous after transpose:", moved.is_contiguous())
try:
    moved.view(2, 2, 6)
except RuntimeError as error:
    print("view() failed:", str(error).split(".")[0])
print("reshape() works:", tuple(moved.reshape(2, 2, 6).shape),
      "| contiguous().view() works:", tuple(moved.contiguous().view(2, 2, 6).shape))

# reshape() is not transpose(): it keeps element order, transpose() changes it.
m = torch.arange(6).reshape(2, 3)
print("\nm            ", m.tolist())
print("m.reshape(3,2)", m.reshape(3, 2).tolist())
print("m.transpose   ", m.transpose(0, 1).tolist())
```

Observed output:

```text
original       (12,) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
reshape(3, 4) 
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
reshape(2, -1) (2, 6) (-1 means: work out this size)
reshape(5, -1) RuntimeError: shape '[5, -1]' is invalid for input of size 12

unsqueeze(0)   (1, 3) (a batch containing one sequence)
unsqueeze(-1)  (3, 1) (each token gets its own row)
squeeze()      (3,)

split features  (2, 3, 2, 2) -> transpose(1, 2) -> (2, 2, 3, 2)
is_contiguous after transpose: False
view() failed: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces)
reshape() works: (2, 2, 6) | contiguous().view() works: (2, 2, 6)

m             [[0, 1, 2], [3, 4, 5]]
m.reshape(3,2) [[0, 1], [2, 3], [4, 5]]
m.transpose    [[0, 3], [1, 4], [2, 5]]
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
"""Chapter 3.7: broadcasting, and the silent bug it can cause.

Run from `code/`:  python examples/ch03/broadcasting.py
"""

import torch

scores = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])               # shape (2, 3)
print("scores + 10           ->", (scores + 10).tolist(), "(a single number is used everywhere)")

per_column = torch.tensor([100.0, 200.0, 300.0])       # shape (3,)
print("scores + per_column   ->", (scores + per_column).tolist(), "(added to every row)")

per_row = torch.tensor([[1000.0], [2000.0]])           # shape (2, 1)
print("scores + per_row      ->", (scores + per_row).tolist(), "(added to every column)")

try:
    scores + torch.tensor([1.0, 2.0])                  # shape (2,) does not line up with (2, 3)
except RuntimeError as error:
    print("scores + shape (2,)   -> RuntimeError:", error)

# THE SILENT BUG: a (3,) tensor and a (3, 1) tensor broadcast to (3, 3).
predictions = torch.tensor([1.0, 2.0, 3.0])            # shape (3,)
targets = torch.tensor([[1.0], [2.0], [3.0]])          # shape (3, 1), e.g. from a careless unsqueeze
difference = predictions - targets
print("\npredictions - targets shape:", tuple(difference.shape), "(expected (3,))")
print(difference)
print("mean absolute difference:", difference.abs().mean().item(), "(should be 0.0; no error was raised)")
print("after fixing the shape  :", (predictions - targets.squeeze(-1)).abs().mean().item())
```

Observed output:

```text
scores + 10           -> [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]] (a single number is used everywhere)
scores + per_column   -> [[101.0, 202.0, 303.0], [104.0, 205.0, 306.0]] (added to every row)
scores + per_row      -> [[1001.0, 1002.0, 1003.0], [2004.0, 2005.0, 2006.0]] (added to every column)
scores + shape (2,)   -> RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1

predictions - targets shape: (3, 3) (expected (3,))
tensor([[ 0.,  1.,  2.],
        [-1.,  0.,  1.],
        [-2., -1.,  0.]])
mean absolute difference: 0.8888888955116272 (should be 0.0; no error was raised)
after fixing the shape  : 0.0
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
"""Chapter 3.8: operations along an axis: sum, mean, max, argmax.

Run from `code/`:  python examples/ch03/reductions.py
"""

import torch

# Scores for 5 candidate tokens, for 2 sequences.
scores = torch.tensor([[0.5, 2.5, 0.25, 1.0, -1.0],
                       [1.5, 0.25, 3.0, 0.0, 0.5]])
print("scores shape          :", tuple(scores.shape))
print("sum over everything   :", round(scores.sum().item(), 4))
print("mean along dim=1      :", [round(v, 3) for v in scores.mean(dim=1).tolist()], "-> one value per sequence (rounded)")
print("max along dim=1       :", scores.max(dim=1).values.tolist())
print("argmax along dim=1    :", scores.argmax(dim=1).tolist(), "-> position of the top score = greedy choice")
print("keepdim=True shape    :", tuple(scores.max(dim=1, keepdim=True).values.shape), "(axis kept with size 1)")

# keepdim matters for broadcasting: subtract each row's maximum from that row.
row_max = scores.max(dim=1, keepdim=True).values
print("scores - row max      :", (scores - row_max).tolist())
top = scores.topk(2, dim=1)
print("topk(2) values/indices:", top.values.tolist(), top.indices.tolist())
```

Observed output:

```text
scores shape          : (2, 5)
sum over everything   : 8.5
mean along dim=1      : [0.65, 1.05] -> one value per sequence (rounded)
max along dim=1       : [2.5, 3.0]
argmax along dim=1    : [1, 2] -> position of the top score = greedy choice
keepdim=True shape    : (2, 1) (axis kept with size 1)
scores - row max      : [[-2.0, 0.0, -2.25, -1.5, -3.5], [-1.5, -2.75, 0.0, -3.0, -2.5]]
topk(2) values/indices: [[2.5, 1.0], [3.0, 1.5]] [[1, 3], [2, 0]]
```

- `argmax(dim=1)` returns the *position* of the largest score in each row, not the score itself. For a row of scores over candidate tokens, that position is the token ID of the greedy choice (Chapter 1.7). Generation code in Chapter 17 does exactly this.
- `max(dim=...)` returns both the values and their positions (`.values` and `.indices`). `topk(k, dim=...)` returns the k largest, which is the core of top-k decoding (Chapter 21.5).
- **`keepdim=True`** keeps the reduced axis with size 1 instead of removing it. Removing it would leave `(2,)`, which broadcasts against `(2, 5)` incorrectly or not at all. Keeping it leaves `(2, 1)`, which broadcasts row by row, so "subtract each row's maximum from that row" works. You will see this precise pattern inside implementations of softmax (Chapter 5.7).

---

### 3.9 Batching: adding a batch dimension

Models process a **batch** of examples in one call, for speed (one large operation instead of many small ones) and, during training, for more stable learning (Chapter 6). By convention the batch is axis 0, the **batch dimension**.

File: [`code/examples/ch03/batching.py`](../../code/examples/ch03/batching.py)

```python
"""Chapter 3.9: batching: adding a batch dimension, stack versus cat.

Run from `code/`:  python examples/ch03/batching.py
"""

import torch

a = torch.tensor([5, 9, 2, 5])
b = torch.tensor([5, 1, 4, 0])
c = torch.tensor([8, 9, 2, 6])

batch = torch.stack([a, b, c])               # new axis 0: (3, 4)
print("stack  ->", tuple(batch.shape))
joined = torch.cat([a, b, c])                # same axis, longer: (12,)
print("cat    ->", tuple(joined.shape))

single = a.unsqueeze(0)                      # a batch of one: (1, 4)
print("one example as a batch ->", tuple(single.shape))

try:
    torch.stack([a, torch.tensor([1, 2])])
except RuntimeError as error:
    print("stack of different lengths -> RuntimeError:", error)

# Models process a whole batch with one call. The same operation runs on every
# example independently, and results come back in the same order.
weights = torch.tensor([1, 10, 100, 1000])
print("batch * weights summed per row:", (batch * weights).sum(dim=1).tolist())
print("same as one at a time         :", [(x * weights).sum().item() for x in (a, b, c)])
```

Observed output:

```text
stack  -> (3, 4)
cat    -> (12,)
one example as a batch -> (1, 4)
stack of different lengths -> RuntimeError: stack expects each tensor to be equal size, but got [4] at entry 0 and [2] at entry 1
batch * weights summed per row: [5295, 415, 6298]
same as one at a time         : [5295, 415, 6298]
```

- `torch.stack` adds a new axis and places the tensors along it: three `(4,)` sequences become one `(3, 4)` batch. `torch.cat` joins tensors along an *existing* axis: three `(4,)` become one `(12,)`. Using `cat` where you meant `stack` produces one long sequence instead of a batch.
- `stack` needs equal shapes. Real sentences have different lengths, so they must be padded to a common length first. Exercise 5 builds that, and Chapter 11 develops it fully.
- Batched computation gives the same results as processing each example separately; the batch is a container, not a mixing step. (Chapter 13 introduces masks precisely to keep padded positions from mixing in.)
- A single example given to a model that expects a batch needs `unsqueeze(0)`. Forgetting it is one of the most common shape errors.

---

### 3.10 Moving between NumPy and PyTorch

File: [`code/examples/ch03/numpy_torch.py`](../../code/examples/ch03/numpy_torch.py)

```python
"""Chapter 3.10: moving between NumPy and PyTorch.

Run from `code/`:  python examples/ch03/numpy_torch.py
"""

import numpy as np
import torch

array = np.array([1.0, 2.0, 3.0])
print("NumPy default float dtype :", array.dtype)

shared = torch.from_numpy(array)             # shares memory with the array
copied = torch.tensor(array)                 # makes a copy
array[0] = 100.0
print("from_numpy after change   :", shared.tolist(), "(shares memory)")
print("torch.tensor after change :", copied.tolist(), "(independent copy)")
print("dtype carried over        :", shared.dtype, "<- float64, not PyTorch's usual float32")
print("convert explicitly        :", torch.from_numpy(array).to(torch.float32).dtype)

back = torch.tensor([[1, 2], [3, 4]]).numpy()
print("tensor.numpy()            :", type(back).__name__, back.dtype, back.shape)
print("tensor.tolist()           :", torch.tensor([[1, 2], [3, 4]]).tolist(), "(plain Python lists)")
```

Observed output:

```text
NumPy default float dtype : float64
from_numpy after change   : [100.0, 2.0, 3.0] (shares memory)
torch.tensor after change : [1.0, 2.0, 3.0] (independent copy)
dtype carried over        : torch.float64 <- float64, not PyTorch's usual float32
convert explicitly        : torch.float32
tensor.numpy()            : ndarray int64 (2, 2)
tensor.tolist()           : [[1, 2], [3, 4]] (plain Python lists)
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
"""Choose where tensors live and measure how much memory they use (Chapter 3).

Every script in the book that uses PyTorch accepts `--device`:

    auto  -> cuda if an NVIDIA GPU is usable, else mps (Apple GPU), else cpu
    cpu   -> always the CPU
    cuda  -> NVIDIA GPU (error if unavailable)
    mps   -> Apple GPU (error if unavailable)
"""

from __future__ import annotations

import argparse

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def available_devices() -> list[str]:
    """Names of the device types PyTorch can use on this machine, CPU always included."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def pick_device(preference: str = "auto") -> torch.device:
    """Turn a --device choice into a torch.device, failing clearly if it is unavailable."""
    if preference not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device {preference!r}; choose one of {DEVICE_CHOICES}")
    available = available_devices()
    if preference == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if candidate in available:
                return torch.device(candidate)
    if preference not in available:
        raise RuntimeError(
            f"Device {preference!r} was requested but is not available here. "
            f"Available: {available}. Use --device cpu or --device auto."
        )
    return torch.device(preference)


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add the book's standard --device option to a command-line parser."""
    parser.add_argument(
        "--device",
        default="auto",
        choices=DEVICE_CHOICES,
        help="where to run: auto (default), cpu, cuda, or mps",
    )


def describe_device(device: torch.device) -> str:
    """A one-line human-readable description of a device."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return (
            f"cuda:{index} {torch.cuda.get_device_name(index)} "
            f"({format_bytes(free)} free of {format_bytes(total)} VRAM)"
        )
    if device.type == "mps":
        return "mps (Apple GPU, shares system memory)"
    return f"cpu ({torch.get_num_threads()} threads used by PyTorch)"


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Memory used by a tensor's numbers: element count times bytes per element."""
    return tensor.nelement() * tensor.element_size()


def format_bytes(count: int | float) -> str:
    """Format a byte count with binary units: 1 KiB = 1024 bytes, 1 MiB = 1024 KiB, ..."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
```

Every PyTorch script from here on accepts `--device auto|cpu|cuda|mps` via `add_device_argument`, and calls `pick_device`. With `auto`, an NVIDIA GPU is preferred, then an Apple GPU, then the CPU. Asking for a device that is not available is an immediate error with a suggestion, rather than a confusing failure later.

---

### 3.12 Measuring how much memory a tensor uses

A tensor's memory is its number of elements times the bytes per element for its dtype; `tensor_bytes` computes exactly that. This lets you estimate, *before* running anything, whether a model or a batch will fit.

File: [`code/scripts/ch03_tensor_tour.py`](../../code/scripts/ch03_tensor_tour.py)

```python
"""Chapter 3 milestone: devices, memory use, and the cost of moving data.

Run from `code/`:
    python -m scripts.ch03_tensor_tour
    python -m scripts.ch03_tensor_tour --device cpu --size 1024
"""

from __future__ import annotations

import argparse
import time

import torch

from llmfp.devices import (
    add_device_argument,
    available_devices,
    describe_device,
    format_bytes,
    pick_device,
    tensor_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_device_argument(parser)
    parser.add_argument("--size", type=int, default=2048, help="side length of the square test tensors")
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def time_matmul(a: torch.Tensor, b: torch.Tensor, repeats: int) -> float:
    """Average seconds for one matrix multiplication, after one warm-up run."""
    (a @ b).sum().item()  # warm-up; .item() also waits for the device to finish
    start = time.perf_counter()
    for _ in range(repeats):
        (a @ b).sum().item()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    args = parse_args()
    try:
        device = pick_device(args.device)
    except RuntimeError as error:
        raise SystemExit(f"Error: {error}")
    print(f"Available devices: {available_devices()}")
    print(f"Using: {describe_device(device)}")

    # Memory: same shape, different data types.
    print("\nMemory for a tensor of shape (32, 256, 768), e.g. a batch of 32 sequences,")
    print("256 tokens each, 768 numbers per token:")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.empty(32, 256, 768, dtype=dtype)
        print(f"  {str(dtype):<15} {format_bytes(tensor_bytes(t)):>10}")

    # Parameters of a model: count times bytes per number.
    print("\nMemory just to hold the parameters of a model with 124 million parameters:")
    for dtype in (torch.float32, torch.bfloat16):
        bytes_needed = 124_000_000 * torch.empty(0, dtype=dtype).element_size()
        print(f"  {str(dtype):<15} {format_bytes(bytes_needed):>10}")

    # Speed: a large matrix multiplication on the chosen device, and on the CPU.
    a = torch.randn(args.size, args.size)
    b = torch.randn(args.size, args.size)
    cpu_seconds = time_matmul(a, b, args.repeats)
    print(f"\n{args.size}x{args.size} matrix multiplication on cpu: {cpu_seconds * 1000:.1f} ms")
    if device.type != "cpu":
        start = time.perf_counter()
        a_dev, b_dev = a.to(device), b.to(device)
        torch.ones(1, device=device).item()  # wait for the copy to finish
        copy_seconds = time.perf_counter() - start
        device_seconds = time_matmul(a_dev, b_dev, args.repeats)
        print(f"copying both tensors to {device.type}: {copy_seconds * 1000:.1f} ms")
        print(f"same multiplication on {device.type}: {device_seconds * 1000:.1f} ms")

    # Tensors on different devices cannot be combined.
    if device.type != "cpu":
        try:
            a + a.to(device)
        except RuntimeError as error:
            print(f"\nMixing devices fails: {str(error).splitlines()[0]}")
    else:
        print("\n(Only the CPU is available, so the GPU comparison is skipped.)")


if __name__ == "__main__":
    main()
```

Observed output on the test machine (CPU only):

```text
Available devices: ['cpu']
Using: cpu (14 threads used by PyTorch)

Memory for a tensor of shape (32, 256, 768), e.g. a batch of 32 sequences,
256 tokens each, 768 numbers per token:
  torch.float32     24.0 MiB
  torch.float16     12.0 MiB
  torch.bfloat16    12.0 MiB

Memory just to hold the parameters of a model with 124 million parameters:
  torch.float32    473.0 MiB
  torch.bfloat16   236.5 MiB

2048x2048 matrix multiplication on cpu: 80.9 ms

(Only the CPU is available, so the GPU comparison is skipped.)
```

And asking for a GPU that is not there:

```text
Error: Device 'cuda' was requested but is not available here. Available: ['cpu']. Use --device cpu or --device auto.
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
ids:
 tensor([[5, 9, 2, 5, 7],
        [5, 1, 0, 0, 0],
        [8, 9, 2, 0, 0]])
mask:
 tensor([[ True,  True,  True,  True,  True],
        [ True,  True, False, False, False],
        [ True,  True,  True, False, False]])
real tokens per sequence: [5, 2, 3]
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

**Next:** [Chapter 4](ch04-data-experiments-reproducibility.md) makes experiments trustworthy: splitting data so evaluation is honest, recognizing leakage and overfitting, and recording every run so it can be reproduced.
