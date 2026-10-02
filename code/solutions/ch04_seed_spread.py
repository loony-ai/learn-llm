"""Chapter 4, Exercise 3 (suggested solution): how much does the split alone change the result?

Repeats the honest evaluation (deduplicated, hash split) with different salts,
for one context size, and reports the spread of validation accuracy. Nothing
about the model changes between runs; only which sentences land in which split.

Run from `code/`:
    python -m solutions.ch04_seed_spread
    python -m solutions.ch04_seed_spread --context-size 8 --repeats 20
"""

from __future__ import annotations

import argparse
import statistics

from llmfp.counting_eval import evaluate_counting_model
from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig, read_lines
from llmfp.splits import deduplicate, hash_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/tiny/harbor_synth.txt")
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    lines = deduplicate(read_lines(args.data))
    accuracies = []
    for salt in range(args.repeats):
        splits = hash_split(lines, (0.8, 0.1, 0.1), salt=str(salt))
        model = CountingLanguageModel(CountingModelConfig(context_size=args.context_size))
        model.train(splits.train)
        accuracy = evaluate_counting_model(model, splits.validation)["accuracy"]
        accuracies.append(accuracy)
        print(f"salt {salt:>2}: validation sentences={len(splits.validation):>3}  accuracy={accuracy:.1%}")
    print(
        f"\ncontext_size={args.context_size}: lowest {min(accuracies):.1%}, highest {max(accuracies):.1%}, "
        f"middle value (median) {statistics.median(accuracies):.1%}"
    )


if __name__ == "__main__":
    main()
