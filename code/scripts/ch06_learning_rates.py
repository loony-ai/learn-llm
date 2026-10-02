"""Chapter 6.10: the same training run with different learning rates.

Run from `code/`:
    python -m scripts.ch06_learning_rates
    python -m scripts.ch06_learning_rates --optimizer adamw --rates 0.0001 0.001 0.01 0.1 1
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from llmfp.experiment import set_seed
from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data
from llmfp.training_basics import evaluate, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_data = make_band_data(2000, seed=args.seed)
    validation_data = make_band_data(500, seed=args.seed + 1)
    loss_fn = nn.CrossEntropyLoss()
    print(f"optimizer={args.optimizer}, epochs={args.epochs}")
    print(f"{'learning rate':>13} | {'loss after 1 epoch':>18} | {'final train loss':>16} | {'val acc':>7}")
    for rate in args.rates:
        set_seed(args.seed)  # identical starting parameters for every rate
        model = TinyMLP(1, 16, 2)
        optimizer_class = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.AdamW
        optimizer = optimizer_class(model.parameters(), lr=rate)
        history = fit(model, train_data, validation_data, loss_fn, optimizer, args.epochs, 32, args.seed)
        final = history[-1]
        accuracy = evaluate(model, *validation_data, loss_fn)["accuracy"] if math.isfinite(final["train_loss"]) else float("nan")
        note = "" if math.isfinite(final["train_loss"]) else f"  (stopped at epoch {final['epoch']}: loss is not a number)"
        print(f"{rate:>13g} | {history[0]['train_loss']:>18.4f} | {final['train_loss']:>16.4f} | {accuracy:>7.1%}{note}")


if __name__ == "__main__":
    main()
