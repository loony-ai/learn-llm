"""Chapter 6, Exercise 3 (suggested solution): what happens if zero_grad is forgotten.

Run from `code/`:  python -m solutions.ch06_forgot_zero_grad
"""

from __future__ import annotations

import torch
from torch import nn

from llmfp.experiment import set_seed
from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data
from llmfp.training_basics import evaluate, iterate_minibatches


def train(zero_grad: bool, epochs: int = 30, learning_rate: float = 0.01) -> list[float]:
    set_seed(0)
    train_data, validation_data = make_band_data(2000, seed=0), make_band_data(500, seed=1)
    model = TinyMLP(1, 16, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    accuracies = []
    for _ in range(epochs):
        for inputs, targets in iterate_minibatches(*train_data, 32, generator=generator):
            if zero_grad:
                optimizer.zero_grad()
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
        accuracies.append(evaluate(model, *validation_data, loss_fn)["accuracy"])
    return accuracies


def main() -> None:
    for zero_grad in (True, False):
        accuracies = train(zero_grad)
        shown = ", ".join(f"{a:.1%}" for a in accuracies[4::5])
        print(f"zero_grad={str(zero_grad):<5}  val accuracy at epochs 5,10,...,30: {shown}")


if __name__ == "__main__":
    main()
