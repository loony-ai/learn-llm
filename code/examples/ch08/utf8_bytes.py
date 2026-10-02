"""Chapter 8.3: how UTF-8 turns characters into bytes.

Run from `code/`:  python examples/ch08/utf8_bytes.py
"""

for character in ["A", "é", "Ж", "न", "灯", "🐟"]:
    data = character.encode("utf-8")
    print(f"{character!r:>5}  {len(data)} byte(s): {' '.join(f'{b:3d}' for b in data)}")

text = "Café 🐟"
data = text.encode("utf-8")
print(f"\n{text!r}: {len(text)} code points -> {len(data)} bytes -> {list(data)}")
print("decoded back:", data.decode("utf-8"))

# Cutting a byte sequence inside a character leaves an incomplete character.
partial = data[:-1]
try:
    partial.decode("utf-8")
except UnicodeDecodeError as error:
    print("\ncut one byte short -> UnicodeDecodeError:", error.reason)
print("with errors='replace':", repr(partial.decode("utf-8", errors="replace")))

# Not every byte sequence is valid UTF-8.
print("invalid bytes [255, 254] ->", repr(bytes([255, 254]).decode("utf-8", errors="replace")))
