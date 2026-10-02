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
