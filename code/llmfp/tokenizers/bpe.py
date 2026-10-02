"""Byte-level byte-pair encoding (Chapter 9, Project 1).

Training starts from the 256 byte values and repeatedly merges the most frequent
adjacent pair of tokens into a new token, recording each merge in order.
Encoding applies the same merges, in the same order of priority, to new text.

Text is first split into chunks (pre-tokenization) so merges never cross the
boundary between, say, a word and the following punctuation or space. The
pattern is a standard-library approximation of GPT-2's (which needs the
third-party `regex` module for its Unicode classes).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from llmfp.tokenizers.base import Tokenizer, register

# Pre-tokenization pattern, tried left to right at each position:
#   English contractions ('s 't 'd 'm 'll 've 're)
#   an optional space + letters      (letters = word characters that are not digits or "_")
#   an optional space + digits
#   an optional space + other symbols (anything that is not whitespace or a word character, or "_")
#   whitespace not followed by a non-space (so " word" keeps its leading space), or any whitespace
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""

Pair = tuple[int, int]


def pretokenize(text: str, pattern: str = PATTERN) -> list[str]:
    chunks = re.findall(pattern, text)
    if "".join(chunks) != text:  # the pattern must cover every character, or text would be lost
        raise AssertionError("pre-tokenization pattern dropped characters")
    return chunks


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, left to right, with `new_id`."""
    result, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


def _pairs(ids: list[int]) -> Counter[Pair]:
    return Counter(zip(ids, ids[1:]))


def _best_pair(pair_counts: Counter[Pair]) -> Pair | None:
    """Most frequent pair; ties go to the smallest pair, so training is deterministic."""
    candidates = [(count, pair) for pair, count in pair_counts.items() if count > 0]
    if not candidates:
        return None
    top = max(count for count, _ in candidates)
    return min(pair for count, pair in candidates if count == top)


def train_merges(chunk_counts: Counter[str], num_merges: int, min_count: int = 2) -> list[Pair]:
    """Learn up to `num_merges` merges from chunk frequencies. Stops early if no pair occurs `min_count` times.

    Incremental: after each merge, only the chunks containing the merged pair are updated.
    `train_merges_reference` below does the same thing the slow, obvious way; tests check
    that both produce identical merges.
    """
    words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
    frequencies = list(chunk_counts.values())
    pair_counts: Counter[Pair] = Counter()
    where: dict[Pair, set[int]] = defaultdict(set)   # pair -> indexes of words containing it
    for index, (ids, frequency) in enumerate(zip(words, frequencies)):
        for pair, count in _pairs(ids).items():
            pair_counts[pair] += count * frequency
            where[pair].add(index)

    merges: list[Pair] = []
    for step in range(num_merges):
        pair = _best_pair(pair_counts)
        if pair is None or pair_counts[pair] < min_count:
            break
        new_id = 256 + step
        merges.append(pair)
        for index in list(where[pair]):
            old = words[index]
            new = merge_pair(old, pair, new_id)
            frequency = frequencies[index]
            for old_pair, count in _pairs(old).items():
                pair_counts[old_pair] -= count * frequency
                where[old_pair].discard(index)
            for new_pair, count in _pairs(new).items():
                pair_counts[new_pair] += count * frequency
                where[new_pair].add(index)
            words[index] = new
        del pair_counts[pair]
    return merges


def train_merges_reference(chunk_counts: Counter[str], num_merges: int, min_count: int = 2) -> list[Pair]:
    """The plain version: recount every pair in every chunk before each merge."""
    words = {chunk: list(chunk.encode("utf-8")) for chunk in chunk_counts}
    merges: list[Pair] = []
    for step in range(num_merges):
        pair_counts: Counter[Pair] = Counter()
        for chunk, ids in words.items():
            for pair, count in _pairs(ids).items():
                pair_counts[pair] += count * chunk_counts[chunk]
        pair = _best_pair(pair_counts)
        if pair is None or pair_counts[pair] < min_count:
            break
        merges.append(pair)
        words = {chunk: merge_pair(ids, pair, 256 + step) for chunk, ids in words.items()}
    return merges


@register
class BPETokenizer(Tokenizer):
    kind = "bpe"

    def __init__(self, merges: list[Pair], special_tokens: Iterable[str] = (), pattern: str = PATTERN) -> None:
        self.merges = [tuple(pair) for pair in merges]
        self.pattern = pattern
        self.ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(self.merges)}
        # The bytes each token stands for: 256 single bytes, then each merge joins two earlier tokens.
        self.token_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        # Special tokens get the IDs after all merges.
        base = len(self.token_bytes)
        self.special_tokens: dict[str, int] = {name: base + i for i, name in enumerate(special_tokens)}
        self.special_names: dict[int, str] = {i: name for name, i in self.special_tokens.items()}
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(
        cls, text: str, vocab_size: int, special_tokens: Iterable[str] = (), pattern: str = PATTERN, min_count: int = 2
    ) -> "BPETokenizer":
        special_tokens = list(special_tokens)
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")
        chunk_counts = Counter(pretokenize(text, pattern))
        return cls(train_merges(chunk_counts, num_merges, min_count), special_tokens, pattern)

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes) + len(self.special_tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge available in this chunk, then look again.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, len(self.ranks)))
            if pair not in self.ranks:
                break
            ids = merge_pair(ids, pair, 256 + self.ranks[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str, *, allowed_special: Iterable[str] = ()) -> list[int]:
        """Encode text. Special-token strings are treated as ordinary text unless allowed.

        Untrusted text (user input, documents) must never be able to inject control
        tokens just by containing their spelling; Chapter 36 returns to this.
        """
        allowed = [name for name in allowed_special if name in self.special_tokens]
        if allowed:
            splitter = "(" + "|".join(re.escape(name) for name in sorted(allowed, key=len, reverse=True)) + ")"
            parts = re.split(splitter, text)
        else:
            parts = [text]
        ids: list[int] = []
        for part in parts:
            if part in allowed:
                ids.append(self.special_tokens[part])
            elif part:
                for chunk in pretokenize(part, self.pattern):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for i in ids:
            if i in self.special_names:
                pieces.append(self.special_names[i].encode("utf-8"))
            else:
                pieces.append(self.token_bytes[i])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        """A readable form of one token, for inspection (partial characters show as U+FFFD)."""
        if token_id in self.special_names:
            return self.special_names[token_id]
        return self.token_bytes[token_id].decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        return {"merges": [list(pair) for pair in self.merges], "special_tokens": list(self.special_tokens), "pattern": self.pattern}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPETokenizer":
        return cls([tuple(pair) for pair in data["merges"]], data["special_tokens"], data["pattern"])
