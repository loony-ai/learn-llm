"""A counting-based next-word language model (Chapter 1).

This is a teaching model, not a neural network. During training it records
how many times each word followed each short context in the training text.
During inference it uses those stored counts to rank candidate next words,
or to choose one at random in proportion to how often it was seen.

The model has the same life cycle as a large language model:

    configure -> train -> save a checkpoint -> load -> predict -> generate

Only the internals differ. Later chapters replace the count table with a
neural network whose stored numbers are learned rather than counted.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Special markers. They are not words from the text; the model adds them so it
# can learn how sentences begin (START) and where they finish (END).
START = "<start>"
END = "<end>"

# A "word" here is either a run of letters/digits/underscores, or a single
# punctuation character. "dawn." becomes ["dawn", "."].
WORD_PATTERN = re.compile(r"\w+|[^\w\s]")

CHECKPOINT_FORMAT = "llmfp-counting-lm-v1"


@dataclass(frozen=True)
class CountingModelConfig:
    """Hyperparameters: choices made before training that training never changes."""

    context_size: int = 2
    lowercase: bool = True

    def __post_init__(self) -> None:
        if self.context_size < 1:
            raise ValueError(f"context_size must be at least 1, got {self.context_size}")


@dataclass(frozen=True)
class GenerationResult:
    """What `generate` produced and why it stopped."""

    words: list[str]
    stop_reason: str  # "end_marker", "unseen_context", or "max_new_words"

    @property
    def text(self) -> str:
        return join_words(self.words)


def split_into_words(text: str, lowercase: bool) -> list[str]:
    """Turn raw text into a list of words and punctuation marks."""
    if lowercase:
        text = text.lower()
    return WORD_PATTERN.findall(text)


def join_words(words: list[str]) -> str:
    """Turn a list of words back into readable text (inverse of splitting, roughly)."""
    text = " ".join(words)
    # Remove the space that `join` put before punctuation: "dawn ." -> "dawn."
    return re.sub(r" ([^\w\s])", r"\1", text)


def rank_followers(followers: Counter[str]) -> list[tuple[str, int]]:
    """Sort (word, count) pairs: highest count first, ties broken alphabetically."""
    return sorted(followers.items(), key=lambda item: (-item[1], item[0]))


class CountingLanguageModel:
    """Predicts the next word by looking up counts gathered from training text."""

    def __init__(self, config: CountingModelConfig) -> None:
        self.config = config
        # counts[context][next_word] = how many times next_word followed context.
        # A context is a tuple of the previous `context_size` words.
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    # ------------------------------------------------------------------ training
    def train(self, lines: Iterable[str]) -> int:
        """Count every (context, next word) pair in the given lines.

        Each non-empty line is treated as one sentence. Returns the number of
        (context, next word) observations that were counted.
        """
        observations = 0
        for line in lines:
            words = split_into_words(line, self.config.lowercase)
            if not words:
                continue
            padded = [START] * self.config.context_size + words + [END]
            # Slide a window over the sentence. At each position, the window's
            # first `context_size` words are the context and the word right
            # after the window is the "next word" we record.
            for position in range(self.config.context_size, len(padded)):
                context = tuple(padded[position - self.config.context_size : position])
                next_word = padded[position]
                self.counts[context][next_word] += 1
                observations += 1
        return observations

    # ----------------------------------------------------------------- inference
    def context_for(self, words: list[str]) -> tuple[str, ...]:
        """Return the last `context_size` words, padding with START if too short."""
        padded = [START] * self.config.context_size + words
        return tuple(padded[-self.config.context_size :])

    def followers_for(self, words: list[str]) -> Counter[str] | None:
        """Return the counts of words seen after the context of `words`, or None.

        This is the model's single lookup step. Prediction and generation both
        go through it, so a subclass can change how lookup works in one place.
        """
        return self.counts.get(self.context_for(words)) or None

    def next_word_candidates(self, prompt: str, top: int | None = 5) -> list[tuple[str, int]]:
        """Rank possible next words for `prompt` by how often they were seen.

        Returns (word, count) pairs, most frequent first. Ties are broken
        alphabetically so the output is the same on every run. An empty list
        means the model never saw this context during training.
        """
        words = split_into_words(prompt, self.config.lowercase)
        followers = self.followers_for(words)
        if followers is None:
            return []
        ranked = rank_followers(followers)
        return ranked if top is None else ranked[:top]

    def generate(
        self,
        prompt: str,
        max_new_words: int = 20,
        rng: random.Random | None = None,
        greedy: bool = False,
    ) -> GenerationResult:
        """Extend `prompt` one word at a time.

        greedy=True always picks the most frequent next word.
        greedy=False picks at random, where a word seen 3 times is three times
        as likely to be picked as a word seen once.
        """
        rng = rng or random.Random()
        words = split_into_words(prompt, self.config.lowercase)
        for _ in range(max_new_words):
            followers = self.followers_for(words)
            if followers is None:
                return GenerationResult(words, "unseen_context")
            if greedy:
                next_word = rank_followers(followers)[0][0]
            else:
                # Sort first so the same seed gives the same choice on every run.
                options = sorted(followers.items())
                next_word = rng.choices(
                    [word for word, _ in options],
                    weights=[count for _, count in options],
                )[0]
            if next_word == END:
                return GenerationResult(words, "end_marker")
            words.append(next_word)
        return GenerationResult(words, "max_new_words")

    # ------------------------------------------------------------- inspection
    def num_parameters(self) -> int:
        """How many stored numbers (counts) the model holds."""
        return sum(len(followers) for followers in self.counts.values())

    def num_contexts(self) -> int:
        return len(self.counts)

    def vocabulary(self) -> set[str]:
        """Every word the model can ever output (including the END marker)."""
        vocab: set[str] = set()
        for followers in self.counts.values():
            vocab.update(followers)
        return vocab

    # ------------------------------------------------------------- checkpoints
    def save(self, path: str | Path) -> None:
        """Write configuration and counts to a JSON checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {"context": list(context), "next": dict(sorted(followers.items()))}
            for context, followers in sorted(self.counts.items())
        ]
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(self.config),
            "counts": entries,
        }
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CountingLanguageModel":
        """Rebuild a model from a checkpoint written by `save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{path} is not a counting-model checkpoint "
                f"(format={payload.get('format')!r}, expected {CHECKPOINT_FORMAT!r})"
            )
        model = cls(CountingModelConfig(**payload["config"]))
        for entry in payload["counts"]:
            model.counts[tuple(entry["context"])] = Counter(entry["next"])
        return model


def read_lines(path: str | Path) -> list[str]:
    """Read a UTF-8 text file and return its non-empty lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
