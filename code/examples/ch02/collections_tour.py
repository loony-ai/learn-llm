"""Chapter 2.4: the collection types this book uses constantly.

Run from `code/`:  python examples/ch02/collections_tour.py
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace

tokens = ["the", "keeper", "lit", "the", "lamp", "."]

# list: ordered, changeable, allows duplicates. Our token sequences are lists.
print("list      ", tokens, "| length", len(tokens), "| first", tokens[0], "| last two", tokens[-2:])

# tuple: ordered, NOT changeable. Usable as a dictionary key; a list is not.
context = ("the", "keeper")
print("tuple     ", context)
try:
    {["the", "keeper"]: 1}
except TypeError as error:
    print("list as key fails:", error)

# dict: maps keys to values. A vocabulary maps each token to an integer ID.
vocab = {token: index for index, token in enumerate(sorted(set(tokens)))}
print("dict      ", vocab)
print("lookup    ", vocab["lamp"], "| missing with .get:", vocab.get("boats"), "| with default:", vocab.get("boats", -1))

# set: unordered, no duplicates, fast membership tests.
unique = set(tokens)
print("set       ", sorted(unique), "| 'lamp' in set:", "lamp" in unique)

# Counter: a dict that counts. Missing keys count as 0 instead of raising KeyError.
counts = Counter(tokens)
print("Counter   ", counts.most_common(2), "| count of 'boats':", counts["boats"])

# defaultdict: creates a default value the first time a missing key is used.
followers = defaultdict(Counter)
for previous, current in zip(tokens, tokens[1:]):
    followers[previous][current] += 1
print("defaultdict", dict(followers))


# dataclass: a class whose main job is to hold named fields.
@dataclass(frozen=True)  # frozen: fields cannot be changed after creation
class TrainingSettings:
    steps: int = 100
    learning_rate: float = 0.001
    tags: list[str] = field(default_factory=list)  # each instance gets its own new list


settings = TrainingSettings(steps=500)
print("dataclass ", settings)
print("as dict   ", asdict(settings))
print("replace   ", replace(settings, learning_rate=0.01))  # a modified copy
try:
    settings.steps = 1  # type: ignore[misc]
except AttributeError as error:
    print("frozen    ", type(error).__name__, "-", error)
