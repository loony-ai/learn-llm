"""Chapter 2: train the counting model from a TOML config with command-line overrides.

Run from `code/` (after `pip install -e ".[dev]"`):

    python -m scripts.ch02_train_counting
    python -m scripts.ch02_train_counting --set model.context_size=3 --set samples=5
    python -m scripts.ch02_train_counting --config configs/counting-cpu.toml --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field

from llmfp.config import ConfigError, load_config, save_json, to_dict
from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig, read_lines

logger = logging.getLogger("ch02_train_counting")


@dataclass(frozen=True)
class CountingRunConfig:
    """Everything one training run needs. Defaults match configs/counting-cpu.toml."""

    data: str = "data/tiny/harbor.txt"
    checkpoint: str = "runs/ch02/counting_model.json"
    seed: int = 0
    samples: int = 3
    prompt: str = "the keeper"
    model: CountingModelConfig = field(default_factory=CountingModelConfig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the counting model from a config file.")
    parser.add_argument("--config", default="configs/counting-cpu.toml", help="TOML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set model.context_size=3 (repeatable)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    try:
        config = load_config(CountingRunConfig, args.config, args.overrides)
    except (ConfigError, FileNotFoundError) as error:
        raise SystemExit(f"Configuration error: {error}")
    logger.info("Config: %s", config)

    lines = read_lines(config.data)
    logger.debug("First training line: %r", lines[0])
    model = CountingLanguageModel(config.model)
    observations = model.train(lines)
    logger.info(
        "Trained on %d lines: %d observations, %d parameters",
        len(lines), observations, model.num_parameters(),
    )

    model.save(config.checkpoint)
    # Record the exact configuration next to the checkpoint, so the run can be repeated.
    config_path = config.checkpoint.removesuffix(".json") + ".config.json"
    save_json(to_dict(config), config_path)
    logger.info("Saved checkpoint to %s and config to %s", config.checkpoint, config_path)

    rng = random.Random(config.seed)
    for i in range(config.samples):
        result = model.generate(config.prompt, rng=rng)
        print(f"Sample {i + 1}: {result.text}  [{result.stop_reason}]")


if __name__ == "__main__":
    main()
