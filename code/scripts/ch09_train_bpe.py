"""Chapter 9 (Project 1): train a byte-level BPE tokenizer and save it.

Run from `code/`:
    python -m scripts.ch09_train_bpe
    python -m scripts.ch09_train_bpe --set vocab_size=512 --set output=runs/bpe-512.json
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from llmfp.config import ConfigError, load_config
from llmfp.experiment import finish_run, start_run
from llmfp.tokenizers import BPETokenizer, load_tokenizer, save_tokenizer


@dataclass(frozen=True)
class BPEConfig:
    runs_root: str = "runs"
    run_name: str = "ch09-bpe"
    corpus: str = "data/tokenizer/corpus.txt"
    vocab_size: int = 2048
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    min_count: int = 2
    output: str = "data/tokenizer/harbor-bpe-2048.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/bpe-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    try:
        config = load_config(BPEConfig, args.config, args.overrides)
    except (ConfigError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}")

    run_dir = start_run(config.runs_root, config.run_name, config, data_files=[config.corpus])
    text = open(config.corpus, encoding="utf-8").read()
    start = time.perf_counter()
    tokenizer = BPETokenizer.train(text, config.vocab_size, config.special_tokens, min_count=config.min_count)
    seconds = time.perf_counter() - start

    ids = tokenizer.encode(text)
    save_tokenizer(tokenizer, config.output)
    reloaded = load_tokenizer(config.output)
    same = reloaded.encode(text[:5000]) == tokenizer.encode(text[:5000])

    print(f"Run: {run_dir}")
    print(f"Trained on {len(text):,} characters in {seconds:.1f} s: {len(tokenizer.merges)} merges, vocab_size {tokenizer.vocab_size}")
    print(f"Corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens ({len(text.encode('utf-8')) / len(ids):.2f} bytes per token)")
    print(f"Round trip on the whole corpus: {tokenizer.decode(ids) == text}; saved to {config.output}; reload identical: {same}")
    print("First 12 merges:", [tokenizer.token_text(256 + i) for i in range(12)])
    print("Last 12 merges: ", [tokenizer.token_text(256 + len(tokenizer.merges) - 12 + i) for i in range(12)])
    print("Special tokens:", tokenizer.special_tokens)
    finish_run(run_dir, {"merges": len(tokenizer.merges), "vocab_size": tokenizer.vocab_size, "train_seconds": seconds,
                         "corpus_tokens": len(ids)})


if __name__ == "__main__":
    main()
