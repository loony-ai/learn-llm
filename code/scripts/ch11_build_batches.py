"""Chapter 11 milestone: from text files to training batches, and a model trained on them.

Encodes the harbor corpus with the Project 1 tokenizer, packs sentences (as
documents) into fixed-length windows with a separator token, serves them with a
DataLoader, and trains a bigram model that predicts at every position. This is
the data path the transformer in Parts 3 and 4 will use.

Run from `code/`:
    python -m scripts.ch11_build_batches
    python -m scripts.ch11_build_batches --set context_length=64 --set batch_size=8
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from llmfp.config import ConfigError, load_config
from llmfp.counting_lm import read_lines
from llmfp.data import pack_documents, pad_batch
from llmfp.devices import add_device_argument, pick_device
from llmfp.experiment import finish_run, set_seed, start_run
from llmfp.model.bigram import BigramModel
from llmfp.splits import deduplicate, hash_split
from llmfp.tokenizers import load_tokenizer


@dataclass(frozen=True)
class BatchConfig:
    runs_root: str = "runs"
    run_name: str = "ch11-batches"
    data: str = "data/tiny/harbor_synth.txt"
    tokenizer: str = "data/tokenizer/harbor-bpe-2048.json"
    separator: str = "<|endoftext|>"
    seed: int = 0
    context_length: int = 32
    batch_size: int = 16
    learning_rate: float = 0.05
    epochs: int = 15


def sequence_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over every position: flatten (batch, seq, vocab) to (batch * seq, vocab)."""
    logits = model(inputs)
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for windows, in loader:
        windows = windows.to(device)
        total += sequence_loss(model, windows[:, :-1], windows[:, 1:]).item() * windows.shape[0]
        count += windows.shape[0]
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/batches-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    add_device_argument(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        config = load_config(BatchConfig, args.config, args.overrides)
        device = pick_device(args.device)
    except (ConfigError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")
    set_seed(config.seed)
    run_dir = start_run(config.runs_root, config.run_name, config, data_files=[config.data, config.tokenizer])

    # 1. Text -> documents of token IDs (each sentence is a document here).
    tokenizer = load_tokenizer(config.tokenizer)
    separator = tokenizer.special_tokens[config.separator]
    splits = hash_split(deduplicate(read_lines(config.data)), salt=str(config.seed))
    train_docs = [tokenizer.encode(line) for line in splits.train]
    val_docs = [tokenizer.encode(line) for line in splits.validation]
    print(f"Run: {run_dir}")
    print(f"Documents: {len(train_docs)} training, {len(val_docs)} validation; "
          f"{sum(map(len, train_docs)):,} training tokens")

    # 2. Padding versus packing.
    padded = pad_batch(train_docs, pad_id=separator)
    waste = 1 - padded.attention_mask.float().mean().item()
    train_windows, train_owners = pack_documents(train_docs, config.context_length, separator)
    val_windows, _ = pack_documents(val_docs, config.context_length, separator)
    print(f"Padding every document to the longest: {tuple(padded.input_ids.shape)}, {waste:.0%} of positions wasted")
    print(f"Packing into windows of {config.context_length} (+1 for the last target): "
          f"{tuple(train_windows.shape)} training, {tuple(val_windows.shape)} validation, no padding")
    crossing = (train_owners[:, 0:1] != train_owners).any(dim=1).float().mean().item()
    print(f"Windows containing more than one document: {crossing:.0%}")

    # 3. DataLoader: shuffled, reproducible batches of windows.
    train_loader = DataLoader(TensorDataset(train_windows), batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(TensorDataset(val_windows), batch_size=64)
    windows, = next(iter(train_loader))
    inputs, targets = windows[:, :-1], windows[:, 1:]
    print(f"One batch: windows {tuple(windows.shape)} -> inputs {tuple(inputs.shape)}, targets {tuple(targets.shape)}")
    print(f"  first input tokens : {[tokenizer.token_text(i) for i in inputs[0, :6].tolist()]}")
    print(f"  first target tokens: {[tokenizer.token_text(i) for i in targets[0, :6].tolist()]}")

    # 4. Train a bigram model: one prediction at every position of every window.
    model = BigramModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print(f"\nLoss reference: even spread over {tokenizer.vocab_size} tokens = {math.log(tokenizer.vocab_size):.3f}")
    print(f"Before training: validation loss {evaluate(model, val_loader, device):.3f}")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        for windows, in train_loader:
            windows = windows.to(device)
            optimizer.zero_grad()
            loss = sequence_loss(model, windows[:, :-1], windows[:, 1:])
            loss.backward()
            optimizer.step()
        history.append(evaluate(model, val_loader, device))
        if epoch in (1, 5, 10, config.epochs):
            print(f"  epoch {epoch:>2}: validation loss {history[-1]:.3f}")

    # 5. What did it learn? The most likely next token after a few tokens.
    model.eval()
    with torch.no_grad():
        for text in [" keeper", " at", "<|endoftext|>"]:
            token = tokenizer.encode(text, allowed_special={config.separator})
            best = model(torch.tensor([token], device=device))[0, -1].topk(3).indices.tolist()
            print(f"  after {text!r:<16} most likely next: {[tokenizer.token_text(i) for i in best]}")
    finish_run(run_dir, {"validation_loss": history, "padding_waste": waste})


if __name__ == "__main__":
    main()
