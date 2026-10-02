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
