"""Chapter 6 milestone: train a network to find the band that Chapter 5 built by hand.

Run from `code/`:
    python -m scripts.ch06_train_band
    python -m scripts.ch06_train_band --set optimizer=sgd --set learning_rate=0.5
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import torch
from torch import nn

from llmfp.config import ConfigError, load_config
from llmfp.experiment import finish_run, set_seed, start_run
from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data
from llmfp.training_basics import evaluate, fit


@dataclass(frozen=True)
class BandConfig:
    runs_root: str = "runs"
    run_name: str = "ch06-band"
    seed: int = 0
    train_examples: int = 2000
    validation_examples: int = 500
    hidden: int = 16
    activation: str = "relu"
    optimizer: str = "adamw"
    learning_rate: float = 0.01
    epochs: int = 30
    batch_size: int = 32


def make_optimizer(name: str, parameters, learning_rate: float) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/band-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--every", type=int, default=5, help="print every Nth epoch")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        config = load_config(BandConfig, args.config, args.overrides)
    except (ConfigError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}")

    set_seed(config.seed)
    run_dir = start_run(config.runs_root, config.run_name, config)
    train_data = make_band_data(config.train_examples, seed=config.seed)
    validation_data = make_band_data(config.validation_examples, seed=config.seed + 1)

    model = TinyMLP(1, config.hidden, 2, activation=config.activation)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config.optimizer, model.parameters(), config.learning_rate)

    # Two reference points before training: the untrained model, and the simplest
    # possible rule (always answer the most common label).
    majority = int(train_data[1].float().mean() < 0.5) ^ 1
    baseline_accuracy = (validation_data[1] == majority).float().mean().item()
    before = evaluate(model, *validation_data, loss_fn)
    print(f"Run: {run_dir}")
    print(f"Baseline 'always answer {majority}': val accuracy {baseline_accuracy:.1%}")
    print(f"Untrained model: val loss {before['loss']:.3f}, val accuracy {before['accuracy']:.1%}")

    history = fit(model, train_data, validation_data, loss_fn, optimizer, config.epochs, config.batch_size, config.seed)
    print(f"{'epoch':>5} | {'train loss':>10} | {'val loss':>8} | {'val acc':>7}")
    for record in history:
        if record["epoch"] % args.every == 0 or record["epoch"] in (1, len(history)):
            print(f"{record['epoch']:>5} | {record['train_loss']:>10.4f} | {record['val_loss']:>8.4f} | {record['val_accuracy']:>7.1%}")

    # What did it learn? Ask for its choice across the input range.
    probe = torch.arange(0.0, 4.01, 0.25).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        choices = model(probe).argmax(dim=-1).tolist()
    print("Learned answer by input (1 = inside band):")
    print("  " + " ".join(f"{x:.2f}" for x in probe.squeeze(-1).tolist()))
    print("  " + " ".join(f"{c:>4}" for c in choices))
    finish_run(run_dir, {"baseline_accuracy": baseline_accuracy, "before": before, "history": history})


if __name__ == "__main__":
    main()
