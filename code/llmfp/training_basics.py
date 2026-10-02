"""A minimal, readable training loop for classification (Chapter 6).

    iterate_minibatches  yield (inputs, targets) batches, shuffled reproducibly
    train_step           one forward pass, loss, backward pass, and parameter update
    evaluate             loss and accuracy with training-only behavior switched off
    fit                  repeat train_step over the data for several epochs, with validation

Chapter 19 replaces this with a full pretraining loop (schedules, clipping,
checkpoints, mixed precision). The core steps stay exactly the same.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def iterate_minibatches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield matching slices of inputs and targets. The last batch may be smaller."""
    if len(inputs) != len(targets):
        raise ValueError(f"{len(inputs)} inputs but {len(targets)} targets")
    order = torch.randperm(len(inputs), generator=generator) if shuffle else torch.arange(len(inputs))
    for start in range(0, len(inputs), batch_size):
        chosen = order[start : start + batch_size]
        yield inputs[chosen], targets[chosen]


def train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One update. Returns the loss measured BEFORE the update."""
    model.train()                      # enable training-only behavior (e.g. dropout)
    optimizer.zero_grad()              # 1. clear gradients left over from the previous step
    logits = model(inputs)             # 2. forward pass: compute scores
    loss = loss_fn(logits, targets)    # 3. measure how wrong the scores are
    loss.backward()                    # 4. backward pass: compute a gradient for every parameter
    optimizer.step()                   # 5. adjust every parameter using its gradient
    return loss.item()


@torch.no_grad()  # decorator form: no gradient bookkeeping anywhere in this function
def evaluate(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, loss_fn: LossFunction, batch_size: int = 1024
) -> dict[str, float]:
    """Average loss and accuracy over a dataset, with the model in evaluation mode."""
    was_training = model.training
    model.eval()                       # disable training-only behavior
    total_loss, correct = 0.0, 0
    for batch_inputs, batch_targets in iterate_minibatches(inputs, targets, batch_size, shuffle=False):
        logits = model(batch_inputs)
        total_loss += loss_fn(logits, batch_targets).item() * len(batch_targets)
        correct += (logits.argmax(dim=-1) == batch_targets).sum().item()
    model.train(was_training)          # restore whatever mode the caller had
    return {"loss": total_loss / len(targets), "accuracy": correct / len(targets)}


def fit(
    model: nn.Module,
    train_data: tuple[torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor],
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Train for `epochs` passes over the training data. Returns one record per epoch.

    Stops early if the training loss becomes NaN or infinite, which means training
    has broken down (Chapter 6.10 shows how a too-large learning rate causes it).
    """
    generator = torch.Generator().manual_seed(seed)  # its own generator: shuffling is reproducible
    history = []
    for epoch in range(1, epochs + 1):
        losses = [
            train_step(model, x, y, loss_fn, optimizer)
            for x, y in iterate_minibatches(*train_data, batch_size, shuffle=True, generator=generator)
        ]
        train_loss = sum(losses) / len(losses)
        validation = evaluate(model, *validation_data, loss_fn)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": validation["loss"], "val_accuracy": validation["accuracy"]}
        history.append(record)
        logger.info("epoch %d: %s", epoch, record)
        if not math.isfinite(train_loss):
            logger.warning("training loss is %s at epoch %d; stopping", train_loss, epoch)
            break
    return history
