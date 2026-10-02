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
