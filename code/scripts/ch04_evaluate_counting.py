"""Chapter 4 milestone: split data honestly, detect leakage, and observe overfitting.

Trains one counting model per context size on the training split and measures it
on the training and validation splits. Every run gets its own directory under
runs/ with its config, environment, data fingerprint, metrics, and log.

Run from `code/`:
    python -m scripts.ch04_evaluate_counting
    python -m scripts.ch04_evaluate_counting --set split=shuffle --set deduplicate=false
    python -m scripts.ch04_evaluate_counting --set evaluate_test=true --set "context_sizes=[2]"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

from llmfp.config import ConfigError, load_config
from llmfp.counting_eval import evaluate_counting_model
from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig, read_lines
from llmfp.experiment import finish_run, set_seed, start_run
from llmfp.splits import count_overlap, deduplicate, hash_split, shuffle_split

logger = logging.getLogger("ch04_evaluate_counting")


@dataclass(frozen=True)
class EvalConfig:
    data: str = "data/tiny/harbor_synth.txt"
    runs_root: str = "runs"
    run_name: str = "ch04-counting-eval"
    split: str = "hash"
    deduplicate: bool = True
    fractions: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 0
    context_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 6, 8, 10])
    evaluate_test: bool = False

    def __post_init__(self) -> None:
        if self.split not in ("hash", "shuffle"):
            raise ValueError(f"split must be 'hash' or 'shuffle', got {self.split!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/counting-eval-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(EvalConfig, args.config, args.overrides)
    except (ConfigError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}")

    set_seed(config.seed)
    run_dir = start_run(config.runs_root, config.run_name, config, data_files=[config.data])
    logger.info("Run directory: %s", run_dir)

    lines = read_lines(config.data)
    unique_count = len(set(lines))
    if config.deduplicate:
        lines = deduplicate(lines)
    if config.split == "hash":
        splits = hash_split(lines, config.fractions, salt=str(config.seed))
    else:
        splits = shuffle_split(lines, config.fractions, seed=config.seed)

    leaked = count_overlap(splits.train, splits.validation)
    print(f"Run: {run_dir}")
    print(f"Data: {config.data} ({unique_count} distinct sentences)")
    print(f"Split: {config.split}, deduplicate={config.deduplicate}, sizes={splits.sizes()}")
    print(f"Validation sentences that also appear in training: {leaked} of {len(splits.validation)}")

    header = f"{'context':>7} | {'params':>6} | {'train acc':>9} | {'val acc':>7} | {'val coverage':>12}"
    if config.evaluate_test:
        header += f" | {'test acc':>8}"
    print(header)
    results = []
    for size in config.context_sizes:
        model = CountingLanguageModel(CountingModelConfig(context_size=size))
        model.train(splits.train)
        train_metrics = evaluate_counting_model(model, splits.train)
        val_metrics = evaluate_counting_model(model, splits.validation)
        row = {
            "context_size": size,
            "parameters": model.num_parameters(),
            "train": train_metrics,
            "validation": val_metrics,
        }
        line = (
            f"{size:>7} | {model.num_parameters():>6} | {train_metrics['accuracy']:>9.1%} | "
            f"{val_metrics['accuracy']:>7.1%} | {val_metrics['coverage']:>12.1%}"
        )
        if config.evaluate_test:
            row["test"] = evaluate_counting_model(model, splits.test)
            line += f" | {row['test']['accuracy']:>8.1%}"
        results.append(row)
        print(line)

    finish_run(run_dir, {"sizes": splits.sizes(), "validation_in_train": leaked, "results": results})


if __name__ == "__main__":
    main()
