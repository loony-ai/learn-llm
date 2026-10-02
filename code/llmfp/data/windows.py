"""Fixed-length training windows over a long sequence of token IDs (Chapter 11.2-11.4).

A window of `context_length` inputs is paired with targets shifted by one:
the target at every position is the token that comes next. One window therefore
holds `context_length` training examples, one per position.

    ids:      t0 t1 t2 t3 t4 t5 t6 ...
    input:    t0 t1 t2 t3            (context_length = 4)
    target:   t1 t2 t3 t4
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset):
    def __init__(self, ids: list[int] | torch.Tensor, context_length: int, stride: int | None = None) -> None:
        self.ids = torch.as_tensor(ids, dtype=torch.int64)
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length   # default: windows do not overlap
        if context_length < 1 or self.stride < 1:
            raise ValueError("context_length and stride must be at least 1")
        # A window starting at `start` needs ids[start : start + context_length + 1]
        # (one extra token for the last target). Windows that would run off the end are dropped.
        usable = len(self.ids) - (context_length + 1)
        self.count = 0 if usable < 0 else usable // self.stride + 1

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.count:
            raise IndexError(f"window {index} out of range for {self.count} windows")
        start = index * self.stride
        chunk = self.ids[start : start + self.context_length + 1]
        return chunk[:-1], chunk[1:]   # inputs, targets: the same tokens shifted by one
