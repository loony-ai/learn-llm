"""Chapter 10.1: token IDs are labels, not measurements.

Run from `code/`:  python examples/ch10/ids_are_labels.py
"""

from llmfp.tokenizers import load_tokenizer

tokenizer = load_tokenizer("data/tokenizer/harbor-bpe-2048.json")
for word in [" dawn", " dusk", " noon", " night", " keeper", " lamp"]:
    print(f"{word!r:<9} -> ID {tokenizer.encode(word)}")

# IDs reflect the ORDER IN WHICH MERGES WERE LEARNED, which depends on frequency
# in the training corpus. A different corpus, or one more merge early on, renumbers
# everything. Arithmetic on IDs ("dusk minus dawn") therefore means nothing.
dawn, dusk = tokenizer.encode(" dawn")[0], tokenizer.encode(" dusk")[0]
print(f"\nID of ' dusk' minus ID of ' dawn' = {dusk - dawn}: a number with no meaning")
print("The token with the ID halfway between them:", repr(tokenizer.token_text((dawn + dusk) // 2)))
