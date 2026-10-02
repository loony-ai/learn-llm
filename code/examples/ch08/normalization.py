"""Chapter 8.4: text that looks identical but is not, and Unicode normalization.

Run from `code/`:  python examples/ch08/normalization.py
"""

import unicodedata

composed = "café"          # é as one code point
decomposed = "café"       # e followed by a combining accent
print(f"{composed!r} == {decomposed!r}: {composed == decomposed}   lengths {len(composed)} vs {len(decomposed)}")
for form in ("NFC", "NFD"):
    a, b = unicodedata.normalize(form, composed), unicodedata.normalize(form, decomposed)
    print(f"after {form}: equal={a == b}, length={len(a)}")

# NFKC also folds "compatibility" variants into plain forms, which changes meaning more.
for text in ["ﬁsh", "Ｈａｒｂｏｒ", "x²", "①"]:
    print(f"NFKC: {text!r:<12} -> {unicodedata.normalize('NFKC', text)!r}")

# Invisible and look-alike whitespace.
for label, text in {
    "normal space": "the lamp",
    "no-break space": "the lamp",
    "zero-width space": "the​lamp",
}.items():
    print(f"{label:<17} {text!r:<16} split() -> {text.split()}")

# Look-alike letters from different scripts.
latin, cyrillic = "harbor", "hаrbor"       # the second 'a' is Cyrillic
print(f"\n{latin!r} == {cyrillic!r}: {latin == cyrillic}; second letter: {unicodedata.name(cyrillic[1])}")
print("case folding 'Straße' vs 'STRASSE':", "Straße".lower() == "STRASSE".lower(), "with casefold():", "Straße".casefold() == "STRASSE".casefold())
