"""Split datasets into training, validation, and test sets (Chapter 4).

Two strategies:

    shuffle_split  shuffle with a seeded random generator, then cut.
                   Simple, but identical items can land in different splits.
    hash_split     assign each item by a fingerprint (hash) of a key.
                   The same key always lands in the same split, on every run
                   and on every machine, even as the dataset grows.

Plus helpers to find duplicates and measure overlap between splits.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Splits(Generic[T]):
    train: list[T]
    validation: list[T]
    test: list[T]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)}


def _check_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must be three non-negative numbers adding up to 1, got {list(fractions)}")
    return fractions[0], fractions[1], fractions[2]


def shuffle_split(items: Sequence[T], fractions: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 0) -> Splits[T]:
    """Shuffle a copy of `items` with a seeded generator, then cut it into three parts."""
    train_fraction, validation_fraction, _ = _check_fractions(fractions)
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * train_fraction)
    validation_end = train_end + round(len(shuffled) * validation_fraction)
    return Splits(shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:])


def stable_fraction(key: str, salt: str = "") -> float:
    """Map a string to a number between 0 and 1 that never changes between runs or machines.

    Python's built-in hash() is deliberately different in every process for
    strings, so it cannot be used for this. SHA-256 always gives the same result.
    """
    digest = hashlib.sha256((salt + key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def hash_split(
    items: Iterable[T],
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    key: Callable[[T], str] = str,
    salt: str = "",
) -> Splits[T]:
    """Assign each item to a split by hashing `key(item)`. Items with equal keys share a split.

    `salt` changes the assignment as a whole (like a seed) while keeping it stable.
    The resulting sizes are close to, but not exactly, the requested fractions.
    """
    train_fraction, validation_fraction, _ = _check_fractions(fractions)
    train: list[T] = []
    validation: list[T] = []
    test: list[T] = []
    for item in items:
        position = stable_fraction(key(item), salt)
        if position < train_fraction:
            train.append(item)
        elif position < train_fraction + validation_fraction:
            validation.append(item)
        else:
            test.append(item)
    return Splits(train, validation, test)


def deduplicate(items: Iterable[T], key: Callable[[T], Hashable] = lambda item: item) -> list[T]:
    """Keep the first occurrence of each item (by `key`), preserving order."""
    seen: set[Hashable] = set()
    unique: list[T] = []
    for item in items:
        marker = key(item)
        if marker not in seen:
            seen.add(marker)
            unique.append(item)
    return unique


def count_overlap(reference: Iterable[T], candidates: Iterable[T]) -> int:
    """How many items of `candidates` also appear in `reference`."""
    reference_set = set(reference)
    return sum(1 for item in candidates if item in reference_set)
