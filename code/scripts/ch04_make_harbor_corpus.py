"""Chapter 4: generate a larger synthetic harbor corpus from templates.

Forty hand-written sentences (Chapter 1) are too few to show overfitting or
leakage convincingly. This script combines actors, actions, and times into a few
thousand sentences. It is deterministic: the same seed always writes the same file.

Because sentences are drawn at random from a limited set of combinations, many
are exact duplicates, just like real scraped text. Section 4.4 uses that.

Run from `code/`:
    python -m scripts.ch04_make_harbor_corpus
    python -m scripts.ch04_make_harbor_corpus --sentences 5000 --seed 1 --output data/tiny/other.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

ACTIONS = {
    "the keeper": [
        "lit the lamp", "cleaned the glass", "wrote in the log", "climbed the stairs",
        "rang the bell", "checked the oil", "fixed the lamp", "watched the boats",
    ],
    "the fishers": [
        "mended the nets", "sold the fish", "left the harbor", "watched the sky",
        "unloaded the boats", "counted the crates",
    ],
    "the children": ["watched the boats", "counted the gulls", "ran along the pier", "fed the gulls"],
    "the harbor master": ["opened the gates", "checked the moorings", "wrote in the log", "warned the fishers"],
    "the gulls": ["followed the boats", "waited on the pier", "circled the market"],
    "the boats": ["left the harbor", "came home", "rocked at the moorings", "waited for the tide"],
}
TIMES = ["at dawn", "at dusk", "at noon", "at night", "before the storm", "after the storm", "in the fog", "in the rain"]


def make_sentence(rng: random.Random) -> str:
    actor = rng.choice(sorted(ACTIONS))
    first, second = rng.sample(ACTIONS[actor], 2)
    time = rng.choice(TIMES)
    style = rng.random()
    if style < 0.4:
        text = f"{actor} {first} {time}"
    elif style < 0.8:
        text = f"{actor} {first} and {second} {time}"
    else:
        text = f"{time} {actor} {first}"
    return text[0].upper() + text[1:] + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sentences", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/tiny/harbor_synth.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lines = [make_sentence(rng) for _ in range(args.sentences)]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} sentences ({len(set(lines))} distinct) to {path}")
    for line in lines[:5]:
        print("  ", line)


if __name__ == "__main__":
    main()
