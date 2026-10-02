"""Chapter 2.7: iterators and generators.

Run from `code/`:  python examples/ch02/generators_tour.py
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator

# An iterable is anything a for-loop can walk over. An iterator is the object
# that does the walking: each next() call returns one item until it is used up.
tokens = ["the", "keeper", "lit"]
iterator = iter(tokens)
print("next:", next(iterator), next(iterator), next(iterator))
try:
    next(iterator)
except StopIteration:
    print("StopIteration: the iterator is used up; the list itself is unchanged:", tokens)


# A generator function uses `yield`. Calling it runs nothing yet; it returns a
# generator, which runs the body only as items are requested.
def numbered_lines(path: str | Path) -> Iterator[tuple[int, str]]:
    print("  (generator started)")
    with open(path, encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):  # reads one line at a time
            line = line.strip()
            if line:
                yield number, line
    print("  (generator finished, file closed)")


lines = numbered_lines("data/tiny/harbor.txt")
print("created:", type(lines).__name__, "- nothing has been read yet")
print("first two:", list(itertools.islice(lines, 2)))  # reads only as much as needed


# Generators let you process files far larger than memory: only the current line
# is held at once. Here we count tokens without storing the file.
total_tokens = sum(len(line.split()) for _, line in numbered_lines("data/tiny/harbor.txt"))
print("total whitespace-separated tokens:", total_tokens)


# THE ONE-SHOT TRAP: an iterator or generator can be consumed only once.
def sentences():
    yield "the keeper lit the lamp"
    yield "the boats left the harbor"


gen = sentences()
first_pass = [s.split()[1] for s in gen]
second_pass = [s.split()[1] for s in gen]  # silently empty: no error, no data
print("first pass:", first_pass, "| second pass:", second_pass)

# The fix, when you need several passes: make a list once (if it fits in memory),
# or call the generator function again for each pass.
materialized = list(sentences())
print("list, pass 1:", len(materialized), "| list, pass 2:", len(materialized))


# Batching a stream: a pattern used for training data in Chapter 11.
def batched(items, size):
    iterator = iter(items)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


print("batches of 3:", list(batched(range(8), 3)))
