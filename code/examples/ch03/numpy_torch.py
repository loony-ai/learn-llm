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
