"""Chapter 2.5: functions, keyword arguments, defaults, and type hints.

Run from `code/`:  python examples/ch02/functions_tour.py
"""

from __future__ import annotations


# Type hints describe what a function expects and returns. Python does not
# enforce them at run time; editors and tools use them to catch mistakes.
def top_tokens(counts: dict[str, int], limit: int = 3) -> list[str]:
    """Return the `limit` most frequent tokens."""
    return sorted(counts, key=counts.get, reverse=True)[:limit]


counts = {"the": 5, "keeper": 2, "lamp": 3, "boats": 1}
print("positional       ", top_tokens(counts, 2))
print("keyword          ", top_tokens(counts, limit=2))
print("default          ", top_tokens(counts))
print("hints not enforced", top_tokens(counts, limit=True))  # True behaves as 1; no error!


# The * makes every following parameter keyword-only. Use it for options whose
# meaning would be unclear as a bare positional value.
def generate(prompt: str, *, max_new_tokens: int = 20, greedy: bool = False) -> str:
    return f"generate({prompt!r}, max_new_tokens={max_new_tokens}, greedy={greedy})"


print("keyword-only      ", generate("the keeper", greedy=True))
try:
    generate("the keeper", 50, True)  # type: ignore[misc]
except TypeError as error:
    print("positional refused", error)


# Functions can return several values as a tuple, unpacked by the caller.
def split_pair(sequence: list[int]) -> tuple[list[int], list[int]]:
    """Inputs are every token but the last; targets are every token but the first."""
    return sequence[:-1], sequence[1:]


inputs, targets = split_pair([10, 11, 12, 13])
print("tuple return      ", inputs, targets)


# A classic bug: a mutable default value is created ONCE, when the function is
# defined, and shared by every call that uses the default.
def remember_bad(token: str, seen: list[str] = []) -> list[str]:  # noqa: B006 (deliberate bug)
    seen.append(token)
    return seen


def remember_good(token: str, seen: list[str] | None = None) -> list[str]:
    seen = [] if seen is None else seen
    seen.append(token)
    return seen


print("mutable default   ", remember_bad("a"), remember_bad("b"))   # second call still sees "a"
print("None default      ", remember_good("a"), remember_good("b"))


# Functions are values: they can be passed to other functions.
def apply_to_all(func, items):
    return [func(item) for item in items]


print("function as value ", apply_to_all(str.upper, ["lit", "lamp"]))
print("lambda            ", apply_to_all(lambda word: len(word), ["lit", "lamp"]))
