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
