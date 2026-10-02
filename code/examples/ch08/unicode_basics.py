"""Chapter 8.2: characters, code points, and what len() actually counts.

Run from `code/`:  python examples/ch08/unicode_basics.py
"""

import unicodedata

for character in ["A", "é", "Ж", "न", "灯", "⚓", "🐟"]:
    code_point = ord(character)                      # the character's number in Unicode
    print(f"{character!r:>5}  U+{code_point:04X}  {code_point:>6}  {unicodedata.name(character)}")

print("\nchr(0x706F) ->", chr(0x706F), "(from number back to character)")

# len() counts code points, which is not always what a reader would call characters.
examples = {
    "plain":                 "lamp",
    "e + combining accent":  "é",              # looks like é, but is two code points
    "precomposed é":         "é",
    "flag (two code points)": "\U0001F1EF\U0001F1F5",  # regional indicators J + P
    "family emoji (ZWJ)":    "\U0001F468‍\U0001F469‍\U0001F467",
}
print()
for label, text in examples.items():
    names = ", ".join(unicodedata.name(c, "?") for c in text)
    print(f"{label:<23} {text!r:<14} len={len(text)}  [{names}]")
