## Chapter 9: Byte-Pair Encoding: Building a Subword Tokenizer (Capstone Project 1)

[Back to index](../../README.md) · Previous: [Chapter 8](ch08-text-unicode-bytes-tokens.md) · Next: [Chapter 10](ch10-embeddings-and-position.md)

Chapter 8 ended with a dilemma. Bytes can represent any text but make sequences long; words keep sequences short but leave much of the input unknown. **Byte-pair encoding** (BPE) resolves it. It starts from bytes, so nothing is ever unknown, and learns from data which byte sequences occur often enough to deserve a token of their own. Frequent words become single tokens, rarer words are split into a few familiar pieces, and anything unusual falls back to bytes.

Byte-level BPE, or close variants of it, is the tokenization used by GPT-2 and by many of the language models that followed. In this chapter you implement it completely, train it, test it, and compare it with GPT-2's published tokenizer. This is the book's first capstone project.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain how BPE training works and carry it out by hand on a small text.
2. Explain why text is pre-tokenized before merging, and what the pre-tokenization pattern decides.
3. Implement BPE training and encoding, and test a fast implementation against a simple reference.
4. Handle special tokens safely, and explain why byte-level BPE needs no unknown token.
5. Save, load, and verify a trained tokenizer.
6. Use a published tokenizer, compare it with your own, and decide when to train a tokenizer and when to reuse one.
7. Measure how tokenization affects different languages, code, numbers, context length, and cost.

#### Prerequisites

- [Chapter 8](ch08-text-unicode-bytes-tokens.md): UTF-8 bytes, the `Tokenizer` interface, round-trip tests, the unit trade-offs.
- [Chapter 4](../part-1-foundations/ch04-data-experiments-reproducibility.md): held-out data, leakage, run records.
- [Chapter 2.10](../part-1-foundations/ch02-python-foundations-and-environment.md): pytest fixtures and parametrization.

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Subword token | A token that may be a whole word or part of one | 9.1 |
| Byte-pair encoding (BPE) | Learning tokens by repeatedly merging the most frequent adjacent pair | 9.2 |
| Merge, merge list | One learned rule "these two tokens become a new token"; the ordered list of all of them | 9.2 |
| Pre-tokenization | Splitting text into chunks (words, spaces, punctuation) before merging, so merges stay inside chunks | 9.3 |
| Rank (of a merge) | Its position in the merge list; earlier merges are applied first when encoding | 9.5 |
| Special token | A reserved token with a control meaning, such as "end of text", that ordinary text must not be able to produce | 9.6 |
| Compression (of a tokenizer) | How much text one token covers on average, here in characters per token | 9.8 |
| Hugging Face Hub | A public site hosting models, tokenizers, and datasets, from which libraries download files | 9.8 |
| Revision (pinned) | A specific saved version of files on the Hub, identified by a commit hash | 9.8 |

---

### 9.1 The problem: characters make sequences too long; words make vocabularies too big

The measurements in Chapter 8.5 framed the problem. What we want is a vocabulary that:

1. **Never fails**: any text, in any script, encodes and decodes exactly.
2. **Is compact**: tens of thousands of tokens at most, so the model's input and output layers stay a manageable size.
3. **Gives short sequences** for the text the model will mostly see, so the context window covers as much text as possible and computation stays affordable.

**Subword tokens** meet all three. Start with the 256 byte values, which guarantees the first property. Then add tokens for the byte sequences that occur most often: common letter pairs, common word pieces, whole common words, a space followed by a common word. Stop when the vocabulary reaches the size you can afford. Text made of common pieces then needs few tokens; rare text costs more, but still works.

The question is which sequences to add. BPE answers it with a simple, greedy rule applied to training text.

---

### 9.2 The BPE idea, worked by hand

The rule: **find the adjacent pair of tokens that occurs most often in the training text, make it a new token, replace every occurrence, and repeat.** Each step is a **merge**, and the ordered list of merges is the tokenizer's entire learned content.

File: [`code/examples/ch09/bpe_by_hand.py`](../../code/examples/ch09/bpe_by_hand.py)

```python
"""Chapter 9.2: byte-pair encoding training, one merge at a time, on a tiny text.

Run from `code/`:  python examples/ch09/bpe_by_hand.py
"""

from collections import Counter

from llmfp.tokenizers.bpe import merge_pair, pretokenize

text = "the keeper lit the lamp. the lamp lit the keeper."
chunks = Counter(pretokenize(text))
print("chunks and how often each occurs:", {chunk.replace(" ", "␣"): n for chunk, n in chunks.items()})

# Start: every chunk is a sequence of single-byte tokens. Show bytes as characters for
# readability, with the space shown as "␣" so leading spaces are visible.
names = {b: chr(b) for b in range(256)}
names[ord(" ")] = "␣"
words = {chunk: list(chunk.encode("utf-8")) for chunk in chunks}


def show(ids):
    return "|".join(names[i] for i in ids)


for step in range(6):
    pairs = Counter()
    for chunk, ids in words.items():
        for pair in zip(ids, ids[1:]):
            pairs[pair] += chunks[chunk]
    top = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:3]
    best, count = top[0]
    new_id = 256 + step
    names[new_id] = names[best[0]] + names[best[1]]
    print(f"\nstep {step + 1}: top pairs {[(names[a] + '+' + names[b], c) for (a, b), c in top]}")
    print(f"  merge {names[best[0]]!r} + {names[best[1]]!r} -> new token {new_id} = {names[new_id]!r} (seen {count} times)")
    words = {chunk: merge_pair(ids, best, new_id) for chunk, ids in words.items()}
    print("  chunks now:", "  ".join(show(ids) for ids in words.values()))
```

Observed output (spaces shown as ␣):

```text
chunks and how often each occurs: {'the': 1, '␣keeper': 2, '␣lit': 2, '␣the': 3, '␣lamp': 2, '.': 2}

step 1: top pairs [('␣+l', 4), ('h+e', 4), ('t+h', 4)]
  merge '␣' + 'l' -> new token 256 = '␣l' (seen 4 times)
  chunks now: t|h|e  ␣|k|e|e|p|e|r  ␣l|i|t  ␣|t|h|e  ␣l|a|m|p  .

step 2: top pairs [('h+e', 4), ('t+h', 4), ('␣+t', 3)]
  merge 'h' + 'e' -> new token 257 = 'he' (seen 4 times)
  chunks now: t|he  ␣|k|e|e|p|e|r  ␣l|i|t  ␣|t|he  ␣l|a|m|p  .

step 3: top pairs [('t+he', 4), ('␣+t', 3), ('␣+k', 2)]
  merge 't' + 'he' -> new token 258 = 'the' (seen 4 times)
  chunks now: the  ␣|k|e|e|p|e|r  ␣l|i|t  ␣|the  ␣l|a|m|p  .

step 4: top pairs [('␣+the', 3), ('␣+k', 2), ('a+m', 2)]
  merge '␣' + 'the' -> new token 259 = '␣the' (seen 3 times)
  chunks now: the  ␣|k|e|e|p|e|r  ␣l|i|t  ␣the  ␣l|a|m|p  .

step 5: top pairs [('␣+k', 2), ('a+m', 2), ('e+e', 2)]
  merge '␣' + 'k' -> new token 260 = '␣k' (seen 2 times)
  chunks now: the  ␣k|e|e|p|e|r  ␣l|i|t  ␣the  ␣l|a|m|p  .

step 6: top pairs [('a+m', 2), ('e+e', 2), ('e+p', 2)]
  merge 'a' + 'm' -> new token 261 = 'am' (seen 2 times)
  chunks now: the  ␣k|e|e|p|e|r  ␣l|i|t  ␣the  ␣l|am|p  .
```

Follow the steps:

1. The text is first split into **chunks** (section 9.3), and the program counts how often each chunk occurs: `␣the` three times, `␣keeper` twice, and so on. Every chunk starts as a sequence of single bytes.
2. At each step, every adjacent pair inside every chunk is counted, weighted by how often the chunk occurs. The winner becomes a new token with the next free ID (256, 257, ...). Ties are broken the same way every time (here, by the smallest pair), so training is deterministic.
3. Merges build on earlier merges. Step 2 creates `he`, step 3 joins `t` and `he` into `the`, and step 4 joins `␣` and `the` into `␣the`. After four merges, the most common word in the text, with its leading space, is a single token.
4. The remaining merges pick up the next most common pieces: `␣k`, then `am` (from "lamp").

Run this for a few thousand steps on a large corpus and the vocabulary fills with common words and word pieces. Nothing in the procedure knows anything about language. The tokens are whatever the frequency counts favor, which is why tokens sometimes look odd to a human ("ropy", "eful") and why a tokenizer trained on one kind of text serves other kinds poorly (section 9.9).

---

### 9.3 Pre-tokenization: splitting before merging

Without any splitting, BPE would happily merge across word boundaries: "the" followed by " keeper" is common in the harbor text, so "the keeper" could become one token, and so could "p." from "lamp." Tokens that straddle words and punctuation waste vocabulary on accidental combinations and make tokenization depend on what happens to follow a word.

**Pre-tokenization** prevents this by first splitting text into chunks, with a regular expression, and only merging *within* chunks. Here is the book's pattern, from the full implementation listed in section 9.4:

```python
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
```

At each position in the text, the alternatives are tried in order:

| Alternative | Matches | Example chunks |
|---|---|---|
| `'(?:[sdmt]\|ll\|ve\|re)` | English contractions | `'s`, `'ll` |
| ` ?[^\W\d_]+` | An optional space, then letters (word characters that are not digits or underscore) | `the`, ` keeper` |
| ` ?\d+` | An optional space, then digits | ` 12345` |
| ` ?(?:[^\s\w]\|_)+` | An optional space, then other symbols | `.`, ` (`, `):` |
| `\s+(?!\S)` | Whitespace, except the last space before a non-space | indentation, blank lines |
| `\s+` | Any remaining whitespace | |

Three design choices are visible in it:

- **The space attaches to the following word.** " keeper" is one chunk, so ` keeper` can become one token. Since most words follow a space, this saves a token per word. It also means "keeper" at the start of a line and " keeper" after a space are different tokens.
- **Runs of whitespace leave their last space for the next word**, so four spaces of indentation before `def` become three spaces plus ` def`.
- **Letters, digits, and symbols never share a chunk.** "lamp." can never become one token.

This pattern is a standard-library approximation of the one GPT-2 uses. GPT-2's pattern uses the Unicode classes `\p{L}` (letters) and `\p{N}` (numbers), which Python's built-in `re` module does not support; it needs the third-party `regex` module. The differences are small and occur at the edges of the Unicode range. One consequence shared by both: vowel signs in scripts such as Devanagari are *combining marks*, not letters, so they fall into the "other symbols" group and split Hindi words into several chunks. Python's `\w` does not match them either; the output below shows the effect.

`pretokenize` checks that the chunks joined together give back the original text exactly. If a pattern ever skipped a character, that character would vanish from training and from encoding, so the code refuses to continue rather than lose data silently. A test runs this check on 200 random Unicode strings.

---

### 9.4 Training merges

File: [`code/llmfp/tokenizers/bpe.py`](../../code/llmfp/tokenizers/bpe.py)

```python
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
```

The training code has two implementations of the same algorithm:

- **`train_merges_reference`** is the hand-worked procedure from section 9.2, written directly: before every merge, recount every pair in every chunk. It is easy to check by reading, and slow.
- **`train_merges`** is the one actually used. It counts pairs once, and also records, for each pair, which chunks contain it (`where`). After a merge, only the chunks that contained the merged pair can change, so only they are updated: their old pairs are subtracted from the counts and their new pairs added. On the Project 1 corpus, this trains a 2,048-token vocabulary in a few seconds.

Optimized code is where bugs hide, so a test (`test_fast_training_matches_reference`) trains both versions on several randomly generated texts and requires identical merge lists. **Keep a slow version whose correctness is easy to check for any algorithm you optimize, and test the fast one against it.** You will see the same technique for the transformer's key-value cache in Chapter 17.6.

Other training details:

- **Merges are learned on chunk frequencies, not on the raw text.** The corpus is reduced to a count of each distinct chunk (about 6,200 distinct chunks for 122,000 chunk occurrences in the Project 1 corpus), so a chunk that occurs a thousand times is processed once, with weight 1,000.
- **`min_count`** stops training when the best pair occurs fewer than twice. Merging a pair that occurs once would only memorize one specific spot in the corpus. On a small corpus, training therefore stops before reaching the requested vocabulary size: on the 40-sentence harbor file, asking for 5,000 tokens yields 387, as `test_training_stops_early_when_no_pair_repeats` records.
- **Token bytes.** `token_bytes` lists the bytes each token stands for: the 256 single bytes, then, for each merge, the two parts joined. Decoding is just looking up and joining these bytes.

---

### 9.5 Encoding and decoding with learned merges

To encode new text, apply the learned merges in **the order they were learned**. A merge's position in the merge list is its **rank**. `_encode_chunk` does it chunk by chunk:

1. Start with the chunk's bytes.
2. Among all adjacent pairs in the current sequence, find the one with the lowest rank (learned earliest). If no adjacent pair is in the merge list, stop.
3. Merge every occurrence of that pair and go back to step 2.

Why order matters: merges were learned on top of earlier merges. `the` was created from `t` and `he`, so `he` must exist before `the` can form. Applying merges in learned order reproduces, on new text, the same segmentation training would have produced. Because the same chunk always encodes the same way, results are cached per chunk; text is full of repeated words, so this makes encoding much faster.

Decoding needs no merges at all: look up each token's bytes, join them, and decode the bytes as UTF-8, with the replacement character for anything incomplete (Chapter 8.3).

---

### 9.6 Special tokens, unknown tokens, and why byte-level BPE needs no unknown token

**No unknown token.** Every text is a sequence of bytes, and every byte value is a token. Text the tokenizer has never seen, in a script that never appeared in training, encodes as more, smaller tokens. The round-trip tests on 200 random Unicode strings pass for this reason. This is the main advantage of building BPE on bytes rather than on characters, whose vocabulary is never complete.

**Special tokens** are reserved tokens with a control meaning rather than a textual one. The most common is an end-of-text marker, written here `<|endoftext|>` as in GPT-2, which separates documents in training data (Chapter 11.7). Chat models add more, to mark where messages from different participants begin and end (Chapter 23.5). They play the role Chapter 1's `<start>` and `<end>` played. In this implementation, special tokens get the IDs after all merges.

The design question is: **how does text become a special token?** If encoding turned every occurrence of the characters `<|endoftext|>` into the special token, then any user, document, or web page containing those characters could insert a control signal into the model's input, for example making the model believe a document has ended and a new one has begun. That is a real class of attack (Chapter 36). The book's `encode` therefore treats special-token spellings as **ordinary text by default**, and produces the special token only for names listed in `allowed_special`, which code that builds training data or prompts uses deliberately. The tests check both behaviors.

Libraries make different choices here, and it is worth knowing which one you are using. The comparison in section 9.8 includes the line `special = ["<|endoftext|>"]`. The Hugging Face library's GPT-2 tokenizer turns the literal text into the single special token (5 tokens in total), while ours encodes it as ordinary characters (16 tokens). Worse, that library's `decode` *skips special tokens by default*, so decoding the result gave `special = [""]`: the text was silently deleted. The comparison script passes `skip_special_tokens=False` to avoid that. Section 9.11's exercises have you test this yourself.

---

### 9.7 Saving and loading a tokenizer

A BPE tokenizer is fully described by three things: the merge list, the special tokens, and the pre-tokenization pattern. `to_dict` writes exactly those, and `save_tokenizer` (Chapter 8.7) stores them as tagged JSON; the trained Project 1 tokenizer is a file of about 1,800 merges.

All three are needed. The same merges with a different pattern split text differently and produce different IDs. The same merges with special tokens in a different order give the special tokens different IDs. A model trained with one and used with the other would silently misread its inputs (Chapter 8.6). The tests save, reload, and compare encodings for this reason.

---

### 9.8 Training your own tokenizer versus using an existing one; comparing against a published tokenizer

#### Project 1's corpus and training run

The training corpus is built from text written for this book, so there are no licensing questions: the synthetic harbor sentences, the source files of the Part 1 chapters (English prose with Markdown), and the Python code of the `llmfp` package from Chapters 1–8.

File: [`code/scripts/ch09_build_corpus.py`](../../code/scripts/ch09_build_corpus.py)

```python
"""Chapter 9: assemble a fixed tokenizer-training corpus from text written for this project.

Sources (all original to this book, so there are no licensing questions):
    data/tiny/harbor_synth.txt              generated harbor sentences
    book/part-1-foundations/*.src.md        Part 1 chapter sources (English prose, Markdown)
    code/llmfp/**/*.py (Chapters 1-8 code)  Python source code

The output is committed to the repository, so every reader trains on identical
text even after the book's own files change.

Run from `code/`:  python -m scripts.ch09_build_corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llmfp.experiment import file_sha256

BOOK = Path("../book/part-1-foundations")
CODE_FILES = [
    "llmfp/counting_lm.py", "llmfp/config.py", "llmfp/devices.py", "llmfp/experiment.py",
    "llmfp/splits.py", "llmfp/counting_eval.py", "llmfp/nn_basics.py", "llmfp/training_basics.py",
    "llmfp/toy_data.py", "llmfp/char_model.py", "llmfp/tokenizers/base.py",
    "llmfp/tokenizers/char.py", "llmfp/tokenizers/byte.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/tokenizer/corpus.txt")
    args = parser.parse_args()

    parts = {"harbor": Path("data/tiny/harbor_synth.txt").read_text(encoding="utf-8")}
    parts["book"] = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(BOOK.glob("ch0*.src.md")))
    parts["code"] = "\n\n".join(Path(f).read_text(encoding="utf-8") for f in CODE_FILES)
    corpus = "\n\n".join(parts[name] for name in ("harbor", "book", "code"))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    for name, text in parts.items():
        print(f"{name:<7} {len(text):>8,} characters")
    print(f"total   {len(corpus):>8,} characters -> {output} (sha256 {file_sha256(output)[:16]}...)")


if __name__ == "__main__":
    main()
```

```text
harbor   151,033 characters
book     320,984 characters
code      49,183 characters
total    521,204 characters -> data/tokenizer/corpus.txt (sha256 a6a8d8a4988f5a84...)
```

The corpus file is committed to the repository, so every reader trains on identical text even as the book's own files change. Training:

File: [`code/configs/bpe-cpu.toml`](../../code/configs/bpe-cpu.toml)

```toml
# Chapter 9 (Project 1): train a byte-level BPE tokenizer.
runs_root = "runs"
run_name = "ch09-bpe"
corpus = "data/tokenizer/corpus.txt"
vocab_size = 2048
special_tokens = ["<|endoftext|>"]
min_count = 2
output = "data/tokenizer/harbor-bpe-2048.json"
```

File: [`code/scripts/ch09_train_bpe.py`](../../code/scripts/ch09_train_bpe.py)

```python
"""Chapter 9 (Project 1): train a byte-level BPE tokenizer and save it.

Run from `code/`:
    python -m scripts.ch09_train_bpe
    python -m scripts.ch09_train_bpe --set vocab_size=512 --set output=runs/bpe-512.json
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from llmfp.config import ConfigError, load_config
from llmfp.experiment import finish_run, start_run
from llmfp.tokenizers import BPETokenizer, load_tokenizer, save_tokenizer


@dataclass(frozen=True)
class BPEConfig:
    runs_root: str = "runs"
    run_name: str = "ch09-bpe"
    corpus: str = "data/tokenizer/corpus.txt"
    vocab_size: int = 2048
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    min_count: int = 2
    output: str = "data/tokenizer/harbor-bpe-2048.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/bpe-cpu.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    try:
        config = load_config(BPEConfig, args.config, args.overrides)
    except (ConfigError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}")

    run_dir = start_run(config.runs_root, config.run_name, config, data_files=[config.corpus])
    text = open(config.corpus, encoding="utf-8").read()
    start = time.perf_counter()
    tokenizer = BPETokenizer.train(text, config.vocab_size, config.special_tokens, min_count=config.min_count)
    seconds = time.perf_counter() - start

    ids = tokenizer.encode(text)
    save_tokenizer(tokenizer, config.output)
    reloaded = load_tokenizer(config.output)
    same = reloaded.encode(text[:5000]) == tokenizer.encode(text[:5000])

    print(f"Run: {run_dir}")
    print(f"Trained on {len(text):,} characters in {seconds:.1f} s: {len(tokenizer.merges)} merges, vocab_size {tokenizer.vocab_size}")
    print(f"Corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens ({len(text.encode('utf-8')) / len(ids):.2f} bytes per token)")
    print(f"Round trip on the whole corpus: {tokenizer.decode(ids) == text}; saved to {config.output}; reload identical: {same}")
    print("First 12 merges:", [tokenizer.token_text(256 + i) for i in range(12)])
    print("Last 12 merges: ", [tokenizer.token_text(256 + len(tokenizer.merges) - 12 + i) for i in range(12)])
    print("Special tokens:", tokenizer.special_tokens)
    finish_run(run_dir, {"merges": len(tokenizer.merges), "vocab_size": tokenizer.vocab_size, "train_seconds": seconds,
                         "corpus_tokens": len(ids)})


if __name__ == "__main__":
    main()
```

```bash
python -m scripts.ch09_train_bpe
```

Observed output:

```text
Run: runs/ch09-bpe/20261002-235437
Trained on 521,204 characters in 2.8 s: 1791 merges, vocab_size 2048
Corpus: 521,350 bytes -> 148,187 tokens (3.52 bytes per token)
Round trip on the whole corpus: True; saved to data/tokenizer/harbor-bpe-2048.json; reload identical: True
First 12 merges: ['he', ' t', ' the', ' a', 'in', 'er', 'or', 'at', 'on', ' c', 'ed', ' s']
Last 12 merges:  ['lit', 'over', 'pare', 'rep', 'sy', 'ating', 'ative', ' could', ' band', ' matches', 'ropy', ' bool']
Special tokens: {'<|endoftext|>': 2047}
```

On its own training corpus, the tokenizer averages about 3.5 bytes per token. The first merges are the most common English pieces (`he`, ` the`, `in`, `er`); the last are rarer words and fragments. The trained tokenizer is saved to [`code/data/tokenizer/harbor-bpe-2048.json`](../../code/data/tokenizer/harbor-bpe-2048.json), where Chapters 10 and 11 load it.

#### A published tokenizer

The alternative to training is reusing a tokenizer someone else trained, usually the one that came with a pretrained model. GPT-2's tokenizer has 50,257 tokens: 256 bytes, 50,000 merges learned from a large collection of English web text, and one special token. It is published on the **Hugging Face Hub**, a public site hosting models, tokenizers, and datasets, under the MIT license (checked on the Hub on 2026-10-02).

The `tokenizers` library, from Hugging Face, loads and runs tokenizers quickly (its core is written in Rust). Chapter 9 adds it to the project's pinned dependencies:

```toml
dependencies = [
    "numpy==2.5.3",
    "torch==2.14.1",
    "tokenizers==0.23.2",   # Chapter 9: Hugging Face tokenizers, to compare with published tokenizers
]
```

After changing `pyproject.toml`, reinstall with `python -m pip install -e ".[dev]"` (Chapter 2.3). The lock file and `ch02_check_env.py` were updated to match. Installing `tokenizers` also installs `huggingface_hub`, the library that downloads files from the Hub; it is a transitive dependency recorded in the lock file.

Files on the Hub change over time. To make results reproducible, the comparison script pins a **revision**: the commit hash identifying one exact version of the files, just as a git commit does. The first run downloads the tokenizer (about 1 MB) into a local cache, by default under `~/.cache/huggingface/hub`; later runs read from the cache. The library may print a warning about unauthenticated requests; public files download without an account. Chapter 23.3 covers caching, revisions, and offline use in detail.

#### The comparison

The comparison uses **held-out text** that is not in the training corpus: new harbor sentences generated with a different seed, the Chapter 8 source file (English prose the tokenizer never saw), a Python script from this chapter, and Chapter 8's multilingual sample. Measuring on training text would flatter our tokenizer, which is the tokenizer equivalent of leakage (Chapter 4.4).

File: [`code/scripts/ch09_compare_tokenizers.py`](../../code/scripts/ch09_compare_tokenizers.py)

```python
"""Chapter 9.8-9.9: compare our BPE tokenizer with GPT-2's published tokenizer and with bytes.

GPT-2's tokenizer is downloaded once from the Hugging Face Hub (about 1 MB, MIT
license) at a pinned revision, then read from the local cache.

Run from `code/`:  python -m scripts.ch09_compare_tokenizers
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

from llmfp.counting_lm import read_lines
from llmfp.tokenizers import ByteTokenizer, load_tokenizer

GPT2_REPO = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"   # pinned commit on the Hub (checked 2026-10-02)

SHOWCASE = [
    "The keeper lit the lamp.",
    " tokenization",
    "    def forward(self, x):",
    "Pier 12345 opened in 1987.",
    "Смотритель зажёг лампу.",
    "🐟",
    'special = ["<|endoftext|>"]',
]


def held_out_texts() -> dict[str, str]:
    """Text NOT in the BPE training corpus, so the comparison is fair."""
    return {
        "harbor (new seed)": "\n".join(read_lines("data/tiny/harbor_synth_seed1.txt")),
        "English prose (Ch 8 source)": Path("../book/part-2-text-to-inputs/ch08-text-unicode-bytes-tokens.src.md").read_text(encoding="utf-8"),
        "Python code (Ch 9 script)": Path("scripts/ch09_train_bpe.py").read_text(encoding="utf-8"),
        "multilingual sample": Path("data/tiny/multilingual.txt").read_text(encoding="utf-8"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ours", default="data/tokenizer/harbor-bpe-2048.json")
    args = parser.parse_args()

    ours = load_tokenizer(args.ours)
    gpt2 = HFTokenizer.from_pretrained(GPT2_REPO, revision=GPT2_REVISION)
    byte = ByteTokenizer()
    tokenizers = {
        f"ours ({ours.vocab_size})": (ours.encode, ours.decode),
        # skip_special_tokens=False: by default the library DROPS special tokens when decoding,
        # which silently deletes text such as a literal "<|endoftext|>" (see section 9.6).
        f"GPT-2 ({gpt2.get_vocab_size()})": (lambda t: gpt2.encode(t).ids, lambda ids: gpt2.decode(ids, skip_special_tokens=False)),
        "bytes (256)": (byte.encode, byte.decode),
    }

    print("Characters per token on held-out text (higher means fewer tokens for the same text):")
    names = list(tokenizers)
    print(f"  {'text':<30}{'chars':>7}" + "".join(f"{name:>14}" for name in names))
    for label, text in held_out_texts().items():
        row = f"  {label:<30}{len(text):>7}"
        for encode, decode in tokenizers.values():
            ids = encode(text)
            assert decode(ids) == text, f"round trip failed for {label}"
            row += f"{len(text) / len(ids):>14.2f}"
        print(row)

    print("\nHow each tokenizer splits some examples (tokens separated by |):")
    for text in SHOWCASE:
        print(f"  {text!r}")
        for name, (encode, decode) in tokenizers.items():
            if name.startswith("bytes"):
                continue
            pieces = [decode([i]) for i in encode(text)]
            print(f"    {name:<13} {len(pieces):>2} tokens: {'|'.join(pieces)}")


if __name__ == "__main__":
    main()
```

```bash
python -m scripts.ch04_make_harbor_corpus --seed 1 --output data/tiny/harbor_synth_seed1.txt --sentences 1000
python -m scripts.ch09_compare_tokenizers
```

Observed output:

```text
Characters per token on held-out text (higher means fewer tokens for the same text):
  text                            chars   ours (2048) GPT-2 (50257)   bytes (256)
  harbor (new seed)               51022          4.39          4.11          1.00
  English prose (Ch 8 source)     31102          2.99          3.85          1.00
  Python code (Ch 9 script)        2750          2.54          2.59          1.00
  multilingual sample               252          0.83          1.28          0.65

How each tokenizer splits some examples (tokens separated by |):
  'The keeper lit the lamp.'
    ours (2048)    6 tokens: The| keeper| lit| the| lamp|.
    GPT-2 (50257)  6 tokens: The| keeper| lit| the| lamp|.
  ' tokenization'
    ours (2048)    2 tokens:  token|ization
    GPT-2 (50257)  2 tokens:  token|ization
  '    def forward(self, x):'
    ours (2048)    9 tokens:    | def| forward|(|self|,| |x|):
    GPT-2 (50257) 10 tokens:  | | | def| forward|(|self|,| x|):
  'Pier 12345 opened in 1987.'
    ours (2048)   12 tokens: P|ier| 12|3|4|5| opened| in| 19|8|7|.
    GPT-2 (50257)  8 tokens: P|ier| 123|45| opened| in| 1987|.
  'Смотритель зажёг лампу.'
    ours (2048)   43 tokens: �|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�| |�|�|�|�|�|�|�|�|�|�| |�|�|�|�|�|�|�|�|�|�|.
    GPT-2 (50257) 28 tokens: �|�|м|о|т|р|и|т|е|л|ь| �|�|а|�|�|�|�|�|�| �|�|а|м|�|�|у|.
  '🐟'
    ours (2048)    4 tokens: �|�|�|�
    GPT-2 (50257)  3 tokens: �|�|�
  'special = ["<|endoftext|>"]'
    ours (2048)   16 tokens: s|p|ec|ial| =| [|"|<|||end|o|f|text|||>|"]
    GPT-2 (50257)  5 tokens: special| =| ["|<|endoftext|>|"]
```

The first table measures **compression**: characters per token, so higher means fewer tokens for the same text. Every tokenizer round-tripped every text exactly (the script checks).

- **On harbor text, our 2,048-token vocabulary beats GPT-2's 50,257.** It was trained on text exactly like this. A tokenizer specialized to a domain can be both much smaller and more efficient on that domain.
- **On general English prose, GPT-2 wins clearly.** Its 50,000 merges cover far more English words. Ours splits unfamiliar words into pieces.
- **On Python code, they are about even.** Ours learned from `llmfp`'s code; GPT-2 learned from web text that contains some code, and it handles indentation poorly (see below).
- **On the multilingual sample, ours is worse than one token per character**, close to raw bytes, because its corpus is English. GPT-2 does better, having seen some non-English web text, but still needs more than one token for most non-Latin characters.

The second part shows segmentations. Both tokenizers split the harbor sentence into the same six tokens and `tokenization` into ` token|ization`. They differ in instructive ways:

- **Indentation**: GPT-2 spends a separate token on each leading space; ours learned a multi-space token from Python code.
- **Numbers**: ours splits `12345` and `1987` into small, arbitrary pieces; GPT-2 has learned some multi-digit tokens. Neither splits numbers consistently (section 9.9).
- **Cyrillic and emoji**: shown one token at a time, most tokens decode to �, because each token holds only part of a character's UTF-8 bytes (Chapter 8.3). The full sequence still decodes correctly.
- **Special-token text**: discussed in section 9.6.

#### When to train your own, and when to reuse

| Situation | Usual choice | Why |
|---|---|---|
| Using or adapting a pretrained model | **Use the model's own tokenizer, always** | The model's parameters are indexed by its tokenizer's IDs (Chapter 8.6) |
| Training a new model from scratch on general text | Train one on a large, representative corpus, or reuse a well-tested published one | A tokenizer is cheap to train but hard to change after the model is trained |
| Training from scratch on a narrow domain (this book's Part 4) | Train on representative domain text | Better compression on the domain, as measured here |
| Multilingual or code-heavy applications | Check compression on samples of each language and code style before committing | Poor compression means shorter effective context and higher cost |

The deciding factor is almost always the model: **once a model is trained, its tokenizer is fixed.** Changing it means retraining, or at least substantial adaptation, of the model's input and output layers.

---

### 9.9 How tokenization affects multilingual text, code, numbers, context length, and cost

Every behavior below follows from what you have built.

**Multilingual text.** A tokenizer's merges reflect its training corpus. Languages that were rare in it get few merges and fall back toward bytes, costing several tokens per character. Chapter 8.5 showed that UTF-8 already costs 2–4 bytes for most non-Latin characters; a tokenizer that learned few merges for a script adds little on top. The same message therefore costs more tokens, more compute, more context, and, with per-token pricing, more money in some languages than others. Measure this for every language your application must serve.

**Code.** Code contains whitespace runs, symbols, and long identifiers. Tokenizers that learned from little code spend many tokens on indentation, as GPT-2 does. Tokenizers used for code-capable models typically include merges for common indentation and syntax.

**Numbers.** How digits are grouped depends on which digit sequences were frequent: `1987` may be one token and `12345` two odd pieces. The model then sees numbers as inconsistent groupings of digits, which is one reason (among others) that language models can be unreliable at arithmetic. Some tokenizers deliberately split numbers into single digits for consistency, a design choice you can make in the pre-tokenization pattern (Exercise 5).

**Context length.** A model's context window is a fixed number of tokens (Chapter 11.2). Better compression means more text fits in it.

**Cost and speed.** Model computation, and API prices (Chapter 41.7), scale with token counts. Compression differences of 30% translate directly into cost differences of about 30%.

**Vocabulary size.** A larger vocabulary compresses better, with diminishing returns, and enlarges the model's input and output layers (one row per token, Chapter 5.6). Exercise 3's solution measures the first half of that trade-off:

```bash
python -m solutions.ch09_vocab_sweep
```

Observed output (characters per token on held-out text):

```text
 vocab  merges  train s     harbor      prose       code multiling.
   256       0      0.1       1.00       1.00       1.00       0.65
   512     256      0.4       3.59       1.78       1.52       0.78
  1024     768      0.9       4.39       2.38       2.02       0.81
  2048    1792      2.2       4.39       2.99       2.54       0.83
  4096    3840      4.8       4.39       3.45       2.97       0.84
  6432    6176      7.1       4.39       3.70       3.12       0.85
```

Harbor text stops improving at about 1,000 tokens: the generated corpus contains few distinct words, and by then they are all single tokens. Prose and code keep improving, more slowly with each doubling. The multilingual sample barely improves at all, because the corpus contains almost no non-English text to learn merges from. And the largest request stopped short: asking for 8,192 tokens produced about 6,400, when no pair occurred at least twice any more.

---

### 9.10 Capstone Project 1: build and test a tokenizer

| | |
|---|---|
| **Problem** | Build a byte-level BPE tokenizer that never fails, compresses its domain well, and is safe and reproducible |
| **Required chapters** | 2 (environment, tests), 4 (held-out data, records), 8 (Unicode, bytes, interface) |
| **Hardware** | CPU; training takes a few seconds; the comparison downloads about 1 MB once |
| **Dependencies** | The Chapter 2 environment plus `tokenizers==0.23.2` |
| **Implementation** | [`llmfp/tokenizers/bpe.py`](../../code/llmfp/tokenizers/bpe.py); step-by-step commands in [`code/projects/p1_tokenizer/README.md`](../../code/projects/p1_tokenizer/README.md) |

#### Success criteria

1. **Exact round trip** on tricky text (accents in both forms, multi-code-point emoji, several scripts, tabs and Windows line endings, code) and on 200 random Unicode strings.
2. **Fast training matches the reference** implementation on several generated texts.
3. **Special tokens** are produced only when explicitly allowed; their spelling in ordinary text stays text.
4. **Save, load, and determinism**: retraining gives identical merges; a reloaded tokenizer encodes identically.
5. **Compression is measured on held-out text** and compared with bytes and with a published tokenizer, with the differences explained.

#### Evaluation

The tests in [`code/tests/test_bpe.py`](../../code/tests/test_bpe.py) check criteria 1–4:

```python
"""Tests for the byte-level BPE tokenizer (Chapter 9, Project 1)."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from llmfp.tokenizers import BPETokenizer, load_tokenizer, save_tokenizer
from llmfp.tokenizers.bpe import merge_pair, pretokenize, train_merges, train_merges_reference
from tests.test_tokenizers import TRICKY_TEXTS, random_unicode_text

HARBOR = Path(__file__).resolve().parents[1] / "data" / "tiny" / "harbor.txt"
SAMPLE = HARBOR.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(SAMPLE, vocab_size=300, special_tokens=["<|endoftext|>"])


def test_merge_pair_is_left_to_right_and_non_overlapping():
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]


def test_pretokenize_covers_every_character():
    generator = random.Random(0)
    for _ in range(200):
        text = random_unicode_text(generator, 30)
        assert "".join(pretokenize(text)) == text


def test_pretokenize_keeps_leading_space_with_word():
    assert pretokenize("the keeper's lamp,  lit") == ["the", " keeper", "'s", " lamp", ",", " ", " lit"]


def test_worked_example_first_merges():
    """The merges printed by examples/ch09/bpe_by_hand.py."""
    chunks = Counter(pretokenize("the keeper lit the lamp. the lamp lit the keeper."))
    merges = train_merges(chunks, 6)
    l, h, e, t, space, k, a, m = (ord(c) for c in "lhet kam")
    assert merges == [(space, l), (h, e), (t, 257), (space, 258), (space, k), (a, m)]


@pytest.mark.parametrize("seed", range(5))
def test_fast_training_matches_reference(seed):
    generator = random.Random(seed)
    words = ["lamp", "lamps", "keeper", "keep", "harbor", "the", "lit", "fog", "fish"]
    text = " ".join(generator.choice(words) for _ in range(300))
    chunks = Counter(pretokenize(text))
    assert train_merges(chunks, 40) == train_merges_reference(chunks, 40)


def test_vocab_size_accounting(tokenizer):
    assert tokenizer.vocab_size == 300
    assert len(tokenizer.merges) == 300 - 256 - 1
    assert tokenizer.special_tokens == {"<|endoftext|>": 299}


def test_training_stops_early_when_no_pair_repeats():
    small = BPETokenizer.train(SAMPLE, vocab_size=5000)
    assert small.vocab_size < 5000   # 40 sentences run out of pairs that occur at least twice


@pytest.mark.parametrize("text", TRICKY_TEXTS + [SAMPLE])
def test_round_trip_tricky_text(tokenizer, text):
    assert tokenizer.round_trips(text)


def test_round_trip_random_unicode(tokenizer):
    generator = random.Random(1)
    for _ in range(200):
        assert tokenizer.round_trips(random_unicode_text(generator, 25))


def test_bpe_compresses_text_like_its_training_data(tokenizer):
    # Even 43 merges cut the token count well below one token per byte.
    assert len(tokenizer.encode(SAMPLE)) < len(SAMPLE.encode("utf-8")) * 0.6


def test_special_token_text_is_ordinary_text_by_default(tokenizer):
    text = "end <|endoftext|> here"
    ids = tokenizer.encode(text)
    assert 299 not in ids
    assert tokenizer.decode(ids) == text


def test_special_token_only_when_allowed(tokenizer):
    ids = tokenizer.encode("a<|endoftext|>b", allowed_special={"<|endoftext|>"})
    assert ids.count(299) == 1
    assert tokenizer.decode(ids) == "a<|endoftext|>b"


def test_save_load_round_trip(tmp_path, tokenizer):
    path = tmp_path / "bpe.json"
    save_tokenizer(tokenizer, path)
    loaded = load_tokenizer(path)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode(SAMPLE) == tokenizer.encode(SAMPLE)
    assert loaded.special_tokens == tokenizer.special_tokens


def test_training_is_deterministic():
    first = BPETokenizer.train(SAMPLE, 320)
    second = BPETokenizer.train(SAMPLE, 320)
    assert first.merges == second.merges


def test_vocab_size_too_small_rejected():
    with pytest.raises(ValueError):
        BPETokenizer.train(SAMPLE, 100)


def test_committed_project_tokenizer_loads_and_round_trips():
    path = Path(__file__).resolve().parents[1] / "data" / "tokenizer" / "harbor-bpe-2048.json"
    tokenizer = load_tokenizer(path)
    assert tokenizer.vocab_size == 2048
    assert tokenizer.round_trips("The keeper lit the lamp. Смотритель 🐟\n    def f(x): return x")
```

```bash
pytest tests/test_bpe.py -q
```

```text
.............................                                            [100%]
29 passed in 0.07s
```

Criterion 5 is met by the comparison in section 9.8.

#### Failure cases to understand

- **Poor compression outside the training domain**: the multilingual sample costs more tokens than characters.
- **Inconsistent number splitting**: digits are grouped by frequency, not value.
- **Partial-character tokens**: individual tokens may decode to �; only complete sequences are guaranteed to decode cleanly. Code that displays tokens one at a time must buffer bytes (Chapter 8, Exercise 6).
- **Library differences in special-token handling**: the same text can encode to different structures, and default decoding can delete text.

#### Debugging exercise

Edit `PATTERN` in a copy of `bpe.py` to remove the final `|\s+` alternative. It looks redundant, since the alternative before it already matches whitespace. Run the tests. Which tests catch the change, and what would have happened without the check in `pretokenize`? Acceptance: you find the inputs that break: a single whitespace character other than a plain space, directly before a non-space character, such as the newline in `"line\nnext"` or the tab in `"tab\tx"`. `\s+(?!\S)` cannot match it (the next character is not a space), and no other alternative accepts a newline or tab, so it matches nothing. `pretokenize` raises its assertion, and the round-trip tests on tricky text (which includes tabs and newlines) and on random text fail. Without the check, every such newline and tab would have vanished silently from training and from encoding, and decoded text would lose its line breaks with no error.

#### Reviewer checklist

- [ ] Byte-level base: all 256 byte values are tokens; no unknown token exists.
- [ ] Pre-tokenization covers every character, with a check that refuses to drop text.
- [ ] Training is deterministic (explicit tie-breaking) and the optimized trainer is tested against a reference.
- [ ] Encoding applies merges by rank; decoding joins token bytes with replacement for invalid UTF-8.
- [ ] Special tokens require explicit permission; tests cover both the allowed and the default case.
- [ ] The saved file contains merges, special tokens, and pattern; reload is tested.
- [ ] Training corpus is fixed and fingerprinted (run record); its license is known.
- [ ] Compression is reported on held-out text, by text type, against a baseline and a published tokenizer pinned to a revision.

#### Extensions that require your own decisions

1. **Digit splitting.** Change the pattern so every digit is its own chunk. Measure compression before and after on text with numbers, and argue whether the consistency is worth the cost.
2. **A multilingual corpus.** Add text in other languages (written by you, or from a source whose license you have checked) and retrain. Decide how much of the corpus each language should get, and measure the effect on each language and on English.
3. **The `regex` module.** Install the third-party `regex` package (pin a version you have tested), use GPT-2's original pattern with `\p{L}` and `\p{N}`, and find inputs on which it splits differently from the book's pattern.
4. **Speed.** Profile encoding a large text and decide whether to optimize, and how you would test that the optimization changes nothing.

---

### 9.11 Common mistakes, recap, concept checks, exercises, answers, and checkpoint

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| Decoded text differs from the input | Pre-tokenization pattern drops characters; normalization applied inside the tokenizer; library decode skipping special tokens | Check `"".join(chunks) == text`; keep the tokenizer lossless; decode with special tokens kept |
| Training gives different merges on each run | Ties broken by dictionary or set order | Explicit tie-breaking rule |
| Fewer tokens than the requested vocabulary size | Not enough repeated pairs in a small corpus (`min_count`) | Expected; use more data or a smaller target |
| A model behaves strangely after a tokenizer change | Merges, pattern, or special-token order differ from training | Save all three; never change a trained model's tokenizer |
| User text containing `<|endoftext|>` alters behavior | Special tokens recognized in untrusted text | Allow special tokens only where code inserts them deliberately |
| Token counts much higher than expected for some text | Text type absent from the tokenizer's training corpus | Measure compression per text type; choose or train accordingly |
| Displayed streaming output shows � | Printing tokens one at a time, mid-character | Buffer bytes until characters are complete |

#### Recap

- **BPE** starts from bytes and repeatedly merges the most frequent adjacent pair; the ordered **merge list** is the tokenizer.
- **Pre-tokenization** splits text into chunks first, so merges stay inside words, numbers, symbols, or whitespace runs; the space attaches to the following word.
- Encoding applies merges in learned order (**rank**); decoding joins token bytes.
- A fast trainer that updates only affected chunks was tested against a simple reference: keep a reference for anything you optimize.
- Byte-level BPE needs no unknown token. **Special tokens** must only come from code that inserts them deliberately.
- A tokenizer is saved as merges, special tokens, and pattern; all three are required.
- Our 2,048-token tokenizer beats GPT-2's on its own domain and loses on general text; compression depends on the training corpus, and it determines context use and cost, especially across languages.

#### Concept checks

1. Starting from `"aaab aaab"`, what is the first merge BPE learns? (Pre-tokenization gives the chunks `aaab` and ` aaab`.)
2. Why must merges be applied in the order they were learned when encoding?
3. What does pre-tokenization prevent, and why does the space attach to the following word?
4. Why does byte-level BPE never need an unknown token?
5. Why does the book's `encode` treat `<|endoftext|>` in text as ordinary characters by default?
6. Which three things must be saved to reproduce a BPE tokenizer's encodings exactly?
7. Why does our tokenizer beat GPT-2's on harbor text but lose on English prose?
8. Why is the comparison done on held-out text rather than the training corpus?
9. Give two reasons a larger vocabulary is not always better.
10. Why do individual tokens of Cyrillic text often display as �?
11. Why is it usually impossible to switch a trained model to a better tokenizer?
12. What is the purpose of `train_merges_reference`?

#### Exercises

**Exercise 1 (by hand).** Carry out three BPE merges by hand on the text `"low lower lowest"`, using the book's pre-tokenization and tie-breaking rule (most frequent pair, ties to the smallest pair of byte values). Check your answer with `train_merges`.

**Exercise 2 (inspect the vocabulary).** Load the Project 1 tokenizer and print: the 20 longest tokens, every token that contains a newline, and the tokens for `" keeper"`, `"keeper"`, and `"Keeper"`. Explain why the three spellings of "keeper" encode differently.

**Exercise 3 (vocabulary size).** Train tokenizers of 256, 512, 1,024, 2,048, 4,096, and 8,192 tokens on the Project 1 corpus and measure characters per token on each held-out text. Describe the trend for each text type, and explain any requested size that was not reached.

**Exercise 4 (special-token safety).** Write two tests: one showing that a user message containing `<|endoftext|>` encodes to the same IDs as its characters do individually, and one showing that the special token appears exactly once when code inserts it with `allowed_special`. Then, using the `tokenizers` library and GPT-2's tokenizer, show how the same message encodes and what default decoding returns.

**Exercise 5 (digits).** Make a copy of the pattern in which every digit is its own chunk (replace ` ?\d+` with ` ?\d`), retrain a 2,048-token tokenizer, and compare token counts on `"Pier 12345 opened in 1987."` and on the held-out prose. What did the change cost, and what did it buy?

**Exercise 6 (fairness measurement).** Using GPT-2's tokenizer, measure tokens per character for each line of the multilingual sample. Rank the languages from cheapest to most expensive. If a service charged per token, how much more would the most expensive language cost than English for the same sentence?

#### Suggested answers and acceptance criteria

**Concept checks**

1. Pairs in `aaab` (weight 1) and ` aaab` (weight 1): `a+a` occurs twice in each chunk (positions 1–2 and 2–3, counted as adjacent pairs), so 4 times; `a+b` twice; ` +a` once. The first merge is `a` + `a`.
2. Later merges were learned on top of earlier ones (`the` from `t` + `he`); applying them out of order would produce segmentations training never produced, and different IDs.
3. It prevents merges across word, number, symbol, and whitespace boundaries. Attaching the space to the following word lets a word and its preceding space become one token, saving a token per word.
4. Every text is UTF-8 bytes, and every one of the 256 byte values is a token.
5. Otherwise any text, including untrusted user input or documents, could insert a control token just by containing its spelling.
6. The merge list, the special tokens (in order), and the pre-tokenization pattern.
7. Its merges were learned from harbor text, so harbor words are single tokens; GPT-2's 50,000 merges cover far more general English words, which ours must split.
8. Measuring on training text overstates compression, because the merges were chosen to fit that exact text.
9. Any two: each extra token adds a row to the model's input and output layers (more parameters and memory); compression improves with diminishing returns; rare tokens get few training examples, so the model learns them poorly.
10. Each Cyrillic letter is two UTF-8 bytes, and a token may hold only one of them, which is not valid UTF-8 on its own.
11. The model's input and output layers are indexed by token ID; a new tokenizer assigns different meanings to the IDs, so the model's learned parameters no longer match.
12. To provide a simple implementation, easy to check by reading, against which the optimized trainer is tested.

**Exercise 1.** The chunks are `low`, ` lower`, ` lowest`, each once. The pairs `l+o` and `o+w` each occur 3 times, the most; the tie goes to the smaller pair of byte values, `l` (108) + `o` (111) before `o` (111) + `w` (119), so merge 1 is `lo`. Then `lo+w` occurs 3 times: merge 2 is `low`. Then ` +low` occurs twice: merge 3 is ` low`. Acceptance: your hand result matches `train_merges(Counter(pretokenize("low lower lowest")), 3)`, decoded with `token_bytes`.

**Exercise 2.** Acceptance: you list the requested tokens and explain that `" keeper"`, `"keeper"`, and `"Keeper"` are different byte sequences (a leading space; a capital letter), each with its own merges. Typically the spaced lowercase form is one token, because it is by far the most frequent in the corpus, while the capitalized form at the start of a line is split.

**Exercise 3.** Solution: [`code/solutions/ch09_vocab_sweep.py`](../../code/solutions/ch09_vocab_sweep.py); observed output in section 9.9. Acceptance: you describe harbor text saturating early, prose and code improving with diminishing returns, multilingual text barely moving, and the largest size stopping short because of `min_count`.

**Exercise 4.** The first part is `test_special_token_text_is_ordinary_text_by_default` and `test_special_token_only_when_allowed` in `tests/test_bpe.py`. Acceptance for the second part: you show GPT-2's tokenizer encoding the spelling as the single ID 50256, and default decoding returning the text without it, as in section 9.6.

**Exercise 5.** Acceptance: a before-and-after table. Expect numbers to split into one token per digit (`12345` becomes five tokens), slightly more tokens on number-heavy text and essentially no change on prose, with the merges no longer spent on digit groups available for other pieces. The argument for: every number is represented the same way, which some model designs prefer; against: more tokens for numbers.

**Exercise 6.** Acceptance: a ranked table of tokens per character by line. Expect English, Python code, and the Western European lines near the bottom; Russian, Arabic, Hindi, Chinese, Japanese, and the emoji line near the top, with the most expensive costing several times as many tokens as English for the same meaning. Your answer should note that the sentences are short and illustrative, so measure on larger samples before drawing conclusions for real decisions.

#### Checkpoint: what you can now do independently

You have completed Capstone Project 1. You can now:

- Train a byte-level BPE tokenizer, explain every merge, and encode and decode with it exactly.
- Design and test a pre-tokenization pattern that cannot lose text.
- Implement special tokens safely and recognize unsafe defaults in libraries.
- Test an optimized algorithm against a reference implementation.
- Load a published tokenizer at a pinned revision and compare tokenizers fairly on held-out text.
- Explain, with measurements, how tokenization affects languages, code, numbers, context, and cost.

**Next:** [Chapter 10](ch10-embeddings-and-position.md) turns token IDs into something a network can learn from: embeddings, and a way to represent position.
