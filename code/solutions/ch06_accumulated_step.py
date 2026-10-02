"""Chapter 6, Exercise 4 (suggested solution): one optimizer step from several micro-batches.

Gradient accumulation lets you train with an effective batch larger than fits in
memory: run several smaller batches through forward and backward, letting their
gradients add up, and only then update the parameters once.

Run from `code/`:  python -m solutions.ch06_accumulated_step
"""

from __future__ import annotations

import torch
from torch import nn

from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data
from llmfp.training_basics import LossFunction


def accumulated_train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
    micro_batches: int,
) -> float:
    """One update using `micro_batches` equal chunks. Returns the average loss.

    Each chunk's loss is weighted by its share of the examples, so the summed
    gradient equals the gradient of the average loss over the whole batch, even
    when the batch does not divide evenly.
    """
    model.train()
    optimizer.zero_grad()                         # once, before the micro-batches
    total = 0.0
    for chunk_inputs, chunk_targets in zip(inputs.chunk(micro_batches), targets.chunk(micro_batches)):
        share = len(chunk_targets) / len(targets)
        loss = loss_fn(model(chunk_inputs), chunk_targets) * share
        loss.backward()                           # adds to .grad
        total += loss.item()
    optimizer.step()                              # once, after all micro-batches
    return total


def main() -> None:
    inputs, targets = make_band_data(100, seed=0)
    loss_fn = nn.CrossEntropyLoss()
    results = {}
    for micro_batches in (1, 4, 7):
        torch.manual_seed(0)
        model = TinyMLP(1, 8, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = accumulated_train_step(model, inputs, targets, loss_fn, optimizer, micro_batches)
        results[micro_batches] = model
        print(f"micro_batches={micro_batches}: loss before update {loss:.6f}")
    reference = list(results[1].parameters())
    for micro_batches in (4, 7):
        same = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(reference, results[micro_batches].parameters()))
        print(f"parameters after one step with {micro_batches} micro-batches match 1 full batch: {same}")


if __name__ == "__main__":
    main()
