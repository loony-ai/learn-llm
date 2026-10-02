"""Chapter 11.3: inputs and targets are the same tokens shifted by one, and three ways to get it wrong.

Run from `code/`:  python examples/ch11/shift_by_one.py
"""

from llmfp.tokenizers import load_tokenizer

tokenizer = load_tokenizer("data/tokenizer/harbor-bpe-2048.json")
ids = tokenizer.encode("The keeper lit the lamp at dusk.")
pieces = [tokenizer.token_text(i) for i in ids]
print("tokens:", pieces)

inputs, targets = ids[:-1], ids[1:]
print("\nCorrect: at each position, the target is the NEXT token")
for position, (x, y) in enumerate(zip(inputs, targets)):
    context = "".join(pieces[: position + 1])
    print(f"  position {position}: sees {context!r:<34} -> predict {tokenizer.token_text(y)!r}")

wrong = {
    "no shift (targets = inputs)": (ids[:-1], ids[:-1]),
    "shifted the wrong way": (ids[1:], ids[:-1]),
    "shifted by two": (ids[:-2], ids[2:]),
}
print("\nWrong versions, position 2:")
for label, (x, y) in wrong.items():
    print(f"  {label:<28} sees {''.join(tokenizer.token_text(i) for i in x[:3])!r:<20} -> 'predict' {tokenizer.token_text(y[2])!r}")
