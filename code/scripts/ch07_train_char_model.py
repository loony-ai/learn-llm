"""Chapter 7 (Project 0): train a next-character network and compare it with counting.

Run from `code/`:
    python -m scripts.ch07_train_char_model
    python -m scripts.ch07_train_char_model --set overfit_one_batch=true
    python -m scripts.ch07_train_char_model --set context_size=4 --device cpu
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import torch
from torch import nn

from llmfp.char_model import (
    CharMLP,
    CharModelConfig,
    CharVocabulary,
    counting_baseline,
    load_checkpoint,
    make_examples,
    sample_text,
    save_checkpoint,
)
from llmfp.config import ConfigError, load_config
from llmfp.counting_lm import read_lines
from llmfp.devices import add_device_argument, describe_device, pick_device
from llmfp.experiment import finish_run, set_seed, start_run
from llmfp.nn_basics import count_parameters
from llmfp.splits import deduplicate, hash_split
from llmfp.training_basics import evaluate, fit, train_step

logger = logging.getLogger("ch07_train_char_model")


@dataclass(frozen=True)
class CharRunConfig:
    runs_root: str = "runs"
    run_name: str = "ch07-char-model"
    data: str = "data/tiny/harbor_synth.txt"
    seed: int = 0
    context_size: int = 12
    hidden: int = 128
    learning_rate: float = 0.003
    epochs: int = 8
    batch_size: int = 128
    prompt: str = "The keeper "
    sample_length: int = 60
    samples: int = 3
    overfit_one_batch: bool = False
    overfit_steps: int = 300


def overfit_one_batch(model: nn.Module, contexts: torch.Tensor, targets: torch.Tensor, config: CharRunConfig) -> None:
    """Section 7.9: train on ONE small batch until the loss is near zero."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    batch = contexts[:32], targets[:32]
    print("Overfitting one batch of 32 examples:")
    for step in range(1, config.overfit_steps + 1):
        loss = train_step(model, *batch, loss_fn, optimizer)
        if step == 1 or step % 50 == 0:
            print(f"  step {step:>4}: loss {loss:.4f}")
    final = evaluate(model, *batch, loss_fn)
    print(f"Final on that batch: loss {final['loss']:.4f}, accuracy {final['accuracy']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/char-model-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    add_device_argument(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        config = load_config(CharRunConfig, args.config, args.overrides)
        device = pick_device(args.device)
    except (ConfigError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")

    set_seed(config.seed)
    run_dir = start_run(config.runs_root, config.run_name, config, data_files=[config.data])

    # 1. Data: deduplicate and split by sentence (Chapter 4), then join each split into one text.
    splits = hash_split(deduplicate(read_lines(config.data)), salt=str(config.seed))
    train_text = "\n".join(splits.train) + "\n"
    validation_text = "\n".join(splits.validation) + "\n"
    vocabulary = CharVocabulary.build(train_text)
    train_x, train_y = make_examples(vocabulary.encode(train_text), config.context_size)
    val_x, val_y = make_examples(vocabulary.encode(validation_text), config.context_size)
    print(f"Run: {run_dir}   device: {describe_device(device)}")
    print(f"Vocabulary: {vocabulary.size} characters {''.join(vocabulary.characters)!r}")
    print(f"Examples: {len(train_y):,} training, {len(val_y):,} validation (context {config.context_size} characters)")

    model = CharMLP(CharModelConfig(vocabulary.size, config.context_size, config.hidden)).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    train_x, train_y, val_x, val_y = (t.to(device) for t in (train_x, train_y, val_x, val_y))

    if config.overfit_one_batch:
        overfit_one_batch(model, train_x, train_y, config)
        finish_run(run_dir, {"mode": "overfit_one_batch"})
        return

    # 2. Reference points before training.
    loss_fn = nn.CrossEntropyLoss()
    most_common = torch.bincount(train_y, minlength=vocabulary.size).argmax()
    even_loss = loss_fn(torch.zeros(1, vocabulary.size), torch.tensor([0])).item()
    counting = counting_baseline(train_x.cpu(), train_y.cpu(), val_x.cpu(), val_y.cpu())
    print("\nReference points on validation:")
    print(f"  even spread over {vocabulary.size} characters: loss {even_loss:.3f}")
    print(f"  always {vocabulary.characters[most_common]!r}: accuracy {(val_y == most_common).float().mean():.1%}")
    print(f"  counting model, same context: accuracy {counting['accuracy']:.1%}, coverage {counting['coverage']:.1%}")

    # 3. Train.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = fit(model, (train_x, train_y), (val_x, val_y), loss_fn, optimizer, config.epochs, config.batch_size, config.seed)
    print(f"\n{'epoch':>5} | {'train loss':>10} | {'val loss':>8} | {'val acc':>7}")
    for record in history:
        print(f"{record['epoch']:>5} | {record['train_loss']:>10.4f} | {record['val_loss']:>8.4f} | {record['val_accuracy']:>7.1%}")

    # 4. Compare with counting on contexts it had seen, and on contexts it had not.
    model.eval()
    with torch.no_grad():
        network_correct = (model(val_x).argmax(dim=-1) == val_y).cpu()
    seen = counting["seen"]
    print("\nValidation accuracy by whether the counting model had seen the context:")
    for label, mask in (("seen contexts", seen), ("unseen contexts", ~seen)):
        if mask.any():
            print(
                f"  {label:<16} ({int(mask.sum()):>5} examples): network {network_correct[mask].float().mean():.1%}, "
                f"counting {counting['correct'][mask].float().mean():.1%}"
            )

    # 5. Save, reload, and check that the reloaded model behaves identically.
    checkpoint_dir = run_dir / "checkpoint"
    save_checkpoint(checkpoint_dir, model, vocabulary, {"run": config.run_name})
    reloaded, reloaded_vocabulary = load_checkpoint(checkpoint_dir, device)
    with torch.no_grad():
        identical = torch.equal(model(val_x[:100]), reloaded(val_x[:100]))
    print(f"\nSaved to {checkpoint_dir}; reloaded model gives identical outputs: {identical}")

    # 6. Generate text from the reloaded model.
    generator = torch.Generator().manual_seed(config.seed)
    print(f"Greedy : {sample_text(reloaded, reloaded_vocabulary, config.prompt, config.sample_length, greedy=True)!r}")
    for i in range(config.samples):
        text = sample_text(reloaded, reloaded_vocabulary, config.prompt, config.sample_length, generator)
        print(f"Sample {i + 1}: {text!r}")

    finish_run(run_dir, {"history": history, "counting": {"accuracy": counting["accuracy"], "coverage": counting["coverage"]}})


if __name__ == "__main__":
    main()
