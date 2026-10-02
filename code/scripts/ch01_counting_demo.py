"""Chapter 1 demo: train, inspect, save, reload, and generate with a counting model.

Run from the `code/` directory:

    python -m scripts.ch01_counting_demo
    python -m scripts.ch01_counting_demo --context-size 1 --prompt "the"
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from llmfp.counting_lm import (
    END,
    CountingLanguageModel,
    CountingModelConfig,
    join_words,
    read_lines,
    split_into_words,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/tiny/harbor.txt", help="UTF-8 text file, one sentence per line")
    parser.add_argument("--context-size", type=int, default=2, help="how many previous words the model looks at")
    parser.add_argument("--prompt", default="the keeper", help="text the model should continue")
    parser.add_argument("--samples", type=int, default=3, help="how many random continuations to show")
    parser.add_argument("--max-new-words", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="random seed; same seed gives same samples")
    parser.add_argument("--checkpoint", default="runs/ch01/counting_model.json", help="where to save the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Configure: hyperparameters are fixed before training starts.
    config = CountingModelConfig(context_size=args.context_size)
    model = CountingLanguageModel(config)
    print(f"Hyperparameters: {config}")

    # 2. Train: read the dataset and count (context, next word) pairs.
    lines = read_lines(args.data)
    observations = model.train(lines)
    print(f"Trained on {len(lines)} sentences ({observations} observations) from {args.data}")
    print(f"Stored parameters (counts): {model.num_parameters()}")
    print(f"Distinct contexts seen:     {model.num_contexts()}")
    print(f"Vocabulary size:            {len(model.vocabulary())} (includes {END})")

    # 3. Inspect: what does the model predict right after the prompt?
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Context the model actually uses: {model.context_for(split_into_words(args.prompt, config.lowercase))}")
    candidates = model.next_word_candidates(args.prompt, top=None)
    if not candidates:
        print("The model never saw this context during training, so it has no prediction.")
    else:
        total = sum(count for _, count in candidates)
        print("Next-word candidates (seen N times out of all continuations of this context):")
        for word, count in candidates[:8]:
            print(f"  {word:<10} {count:>3} of {total}")
        if len(candidates) > 8:
            print(f"  ... and {len(candidates) - 8} more")

    # 4. Save a checkpoint and load it back; predictions must not change.
    model.save(args.checkpoint)
    reloaded = CountingLanguageModel.load(args.checkpoint)
    same = reloaded.next_word_candidates(args.prompt, top=None) == candidates
    size_kb = Path(args.checkpoint).stat().st_size / 1024
    print(f"\nSaved checkpoint to {args.checkpoint} ({size_kb:.1f} KB); reloaded predictions identical: {same}")

    # 5. Generate: greedy once, then several random samples.
    greedy = reloaded.generate(args.prompt, max_new_words=args.max_new_words, greedy=True)
    print(f"\nGreedy:   {greedy.text!r}  [stopped: {greedy.stop_reason}]")
    # Normalize training sentences the same way the model does, so we can tell
    # whether a generated sentence is a verbatim copy of one of them.
    training_sentences = {join_words(split_into_words(line, config.lowercase)) for line in lines}
    rng = random.Random(args.seed)
    for i in range(args.samples):
        sample = reloaded.generate(args.prompt, max_new_words=args.max_new_words, rng=rng)
        origin = "copy of a training sentence" if sample.text in training_sentences else "not in training text"
        print(f"Sample {i + 1}: {sample.text!r}  [stopped: {sample.stop_reason}; {origin}]")


if __name__ == "__main__":
    main()
