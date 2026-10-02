## Chapter 8: Text, Unicode, Bytes, and the Need for Tokens

[Back to index](../../README.md) · Previous: [Chapter 7](../part-1-foundations/ch07-project-0-char-model.md) · Next: Chapter 9 (planned)

A model consumes integers. Text is not integers. Every language model therefore starts with a component that converts text into a sequence of integer IDs and back: the **tokenizer**. Chapter 1 split on words and lowercased everything. Chapter 7 used characters and refused any character it had not seen. Both were teaching shortcuts that would fail on real text: other languages, emoji, code, accented letters typed two different ways.

This chapter explains what text actually is inside a computer, from the bottom up, so that the tokenizer you build in Chapter 9 rests on solid ground. You will see why the same visible word can be stored in different ways, why a "character" is harder to define than it looks, how UTF-8 turns any text into bytes, and what is gained and lost by choosing words, characters, or bytes as a model's basic units.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain characters, code points, Unicode, and encodings, and predict how many bytes UTF-8 uses for a character.
2. Recognize text that looks identical but differs underneath, and use normalization appropriately.
3. Compare words, characters, and bytes as model units in terms of vocabulary size, sequence length, and unknown inputs.
4. Define tokens, vocabularies, and token IDs precisely.
5. Implement tokenizers behind a common interface, with saving, loading, and round-trip tests.

#### Prerequisites

- [Chapter 1.6](../part-1-foundations/ch01-what-a-language-model-predicts.md): the informal idea of a token.
- [Chapter 2.8](../part-1-foundations/ch02-python-foundations-and-environment.md): text versus bytes, encodings, why `encoding="utf-8"` is always explicit.
- [Chapter 7.2–7.4](../part-1-foundations/ch07-project-0-char-model.md): the character vocabulary and why IDs are labels, not measurements.

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Unicode | The international standard that assigns a number to every character in every writing system | 8.2 |
| Code point | The number Unicode assigns to one character, written like U+00E9 | 8.2 |
| Combining character | A code point that modifies the one before it, such as an accent | 8.2 |
| Grapheme (user-perceived character) | What a reader sees as one character, which may be several code points | 8.2 |
| Encoding, UTF-8 | A rule turning code points into bytes; UTF-8 is the dominant one, using 1 to 4 bytes per code point | 8.3 |
| Replacement character | U+FFFD (�), substituted for bytes that do not form valid text | 8.3 |
| Normalization (NFC, NFD, NFKC) | Converting text to a standard form so equivalent sequences become identical | 8.4 |
| Token | The unit of text a model reads and predicts, defined by a tokenizer | 8.6 |
| Vocabulary | The fixed set of tokens a tokenizer can produce, each with an ID | 8.6 |
| Token ID | The integer that identifies a token: a label, not a quantity | 8.6 |
| Tokenizer | The component that converts text to token IDs (encode) and back (decode) | 8.6 |
| Unknown token | A reserved ID used for input the vocabulary cannot represent | 8.7 |
| Round trip | Encoding and then decoding; a lossless tokenizer gives back the exact original text | 8.8 |

---

### 8.1 The problem: models consume integers, and text is not integers

The harbor office wants its model to handle messages from visiting crews, some of them in French, Russian, Hindi, or Japanese, plus an occasional emoji and a log line copied from a script. Try the tools from Part 1 on a sample of such text:

- Chapter 1's word splitting knows only the words it saw in training, so almost every non-English word is unknown.
- Chapter 7's character vocabulary has never seen Cyrillic, Devanagari, or Chinese characters, so it can only refuse them, or (with an unknown token) silently replace them.

Section 8.5 measures both failures. To fix them, we need to understand what text is, at a lower level than words or letters.

---

### 8.2 Characters, code points, and Unicode

Computers store numbers. To store text, every character must be given a number. **Unicode** is the international standard that does this for essentially every writing system in use, plus many historical ones and symbols such as emoji. The number assigned to a character is its **code point**, conventionally written as `U+` followed by the number in hexadecimal (base 16): `U+00E9` is é. Python's `ord` gives a character's code point, and `chr` goes the other way.

File: [`code/examples/ch08/unicode_basics.py`](../../code/examples/ch08/unicode_basics.py)

```python
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
```

Observed output:

```text
  'A'  U+0041      65  LATIN CAPITAL LETTER A
  'é'  U+00E9     233  LATIN SMALL LETTER E WITH ACUTE
  'Ж'  U+0416    1046  CYRILLIC CAPITAL LETTER ZHE
  'न'  U+0928    2344  DEVANAGARI LETTER NA
  '灯'  U+706F   28783  CJK UNIFIED IDEOGRAPH-706F
  '⚓'  U+2693    9875  ANCHOR
  '🐟'  U+1F41F  128031  FISH

chr(0x706F) -> 灯 (from number back to character)

plain                   'lamp'         len=4  [LATIN SMALL LETTER L, LATIN SMALL LETTER A, LATIN SMALL LETTER M, LATIN SMALL LETTER P]
e + combining accent    'é'           len=2  [LATIN SMALL LETTER E, COMBINING ACUTE ACCENT]
precomposed é           'é'            len=1  [LATIN SMALL LETTER E WITH ACUTE]
flag (two code points)  '🇯🇵'           len=2  [REGIONAL INDICATOR SYMBOL LETTER J, REGIONAL INDICATOR SYMBOL LETTER P]
family emoji (ZWJ)      '👨\u200d👩\u200d👧' len=5  [MAN, ZERO WIDTH JOINER, WOMAN, ZERO WIDTH JOINER, GIRL]
```

The first table shows the range: Latin letters have small code points; Cyrillic, Devanagari, Chinese characters, and emoji have larger ones. Unicode has room for over a million code points, though far fewer are assigned.

The second table shows why "character" is slippery:

- **`len()` counts code points.** That is usually what you want, but not always what a person sees.
- **Combining characters.** `"é"` is an `e` followed by a *combining* acute accent, which attaches itself to the previous code point. It displays as é but has length 2. The single precomposed code point `U+00E9` also displays as é, with length 1. Section 8.4 deals with this.
- **Sequences that display as one symbol.** A flag is two "regional indicator" code points. The family emoji is five code points: man, woman, girl, joined by invisible *zero-width joiner* characters. A reader sees one symbol: one **grapheme**, or user-perceived character.

For tokenizers, this means a vocabulary built from "characters" is really built from code points, and a single visible symbol can become several tokens. That is normal and harmless as long as decoding puts the code points back together in order.

---

### 8.3 Encodings: how UTF-8 turns characters into bytes

Code points are abstract numbers. Files, network messages, and memory hold **bytes**, each a number from 0 to 255. An **encoding** is the rule for writing code points as bytes. **UTF-8** is by far the most common: it is the default for web pages, source code, and most data files, and it is the encoding this book uses everywhere (Chapter 2.8).

File: [`code/examples/ch08/utf8_bytes.py`](../../code/examples/ch08/utf8_bytes.py)

```python
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
```

Observed output:

```text
  'A'  1 byte(s):  65
  'é'  2 byte(s): 195 169
  'Ж'  2 byte(s): 208 150
  'न'  3 byte(s): 224 164 168
  '灯'  3 byte(s): 231 129 175
  '🐟'  4 byte(s): 240 159 144 159

'Café 🐟': 6 code points -> 10 bytes -> [67, 97, 102, 195, 169, 32, 240, 159, 144, 159]
decoded back: Café 🐟

cut one byte short -> UnicodeDecodeError: unexpected end of data
with errors='replace': 'Café �'
invalid bytes [255, 254] -> '��'
```

How UTF-8 behaves:

- **Variable length.** Basic English letters, digits, and punctuation (the ASCII range, code points below 128) take one byte, identical to the older ASCII encoding. Most European accented letters, Cyrillic, Greek, Arabic, and Hebrew take two bytes. Most other scripts, including Devanagari and Chinese, take three. Emoji and rarer characters take four.
- **Self-describing bytes.** The first byte of a multi-byte character signals how many bytes follow, and the following bytes have a distinctive form. Software can always tell where characters begin, even in the middle of a stream.
- **Not every byte sequence is valid text.** Cut a four-byte emoji after three bytes and decoding fails ("unexpected end of data"). Bytes such as 255 never appear in valid UTF-8. With `errors="replace"`, Python substitutes the **replacement character** U+FFFD (shown as �) for anything invalid instead of raising an error.

That last point matters in practice. A model that generates bytes or byte-based tokens one at a time can produce the first byte of an emoji in one step and the rest in later steps. Code that decodes and displays text after every step (streaming, Chapter 39.6) will briefly show � unless it waits for complete characters. You will also see � in model output when a model generates a byte sequence that is not valid UTF-8 at all.

---

### 8.4 Normalization and invisible differences

Two strings can look identical on screen and still be different sequences of code points. For a tokenizer, they become different token sequences, and for a deduplication step, different "documents".

File: [`code/examples/ch08/normalization.py`](../../code/examples/ch08/normalization.py)

```python
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
```

Observed output:

```text
'café' == 'café': False   lengths 4 vs 5
after NFC: equal=True, length=4
after NFD: equal=True, length=5
NFKC: 'ﬁsh'        -> 'fish'
NFKC: 'Ｈａｒｂｏｒ'     -> 'Harbor'
NFKC: 'x²'         -> 'x2'
NFKC: '①'          -> '1'
normal space      'the lamp'       split() -> ['the', 'lamp']
no-break space    'the\xa0lamp'    split() -> ['the', 'lamp']
zero-width space  'the\u200blamp'  split() -> ['the\u200blamp']

'harbor' == 'hаrbor': False; second letter: CYRILLIC SMALL LETTER A
case folding 'Straße' vs 'STRASSE': False with casefold(): True
```

What each part shows:

- **Composed and decomposed forms.** `café` with a precomposed é and `café` with e plus a combining accent are not equal. **Unicode normalization** converts text to a standard form. **NFC** (normalization form C, "composed") combines sequences into single code points where possible; **NFD** ("decomposed") splits them apart. After either one, the two versions are equal. Text typed on different systems, or copied from different sources, can arrive in either form.
- **Compatibility forms.** **NFKC** also replaces "compatibility" characters with plain equivalents: the `ﬁ` ligature becomes `fi`, full-width letters become ordinary ones, `²` becomes `2`, and `①` becomes `1`. This is more aggressive. It can help matching and search, but it changes meaning in some contexts (superscripts in formulas, deliberate styling), so it is not applied blindly.
- **Look-alike whitespace.** A no-break space looks like a space, and Python's `split()` treats it as one. A zero-width space is invisible, and `split()` does not treat it as whitespace, so "the​lamp" stays one word. Zero-width characters are a classic way to make text look normal while defeating filters, which Chapter 36 revisits under security.
- **Look-alike letters.** The Cyrillic `а` is visually identical to the Latin `a` but is a different code point. Normalization does not change it, because it is a different letter, not a different form of the same one.
- **Case.** `lower()` does not make "Straße" equal to "STRASSE"; `casefold()`, a more thorough case removal designed for comparisons, does.

**What tokenizers do with this.** Practice varies, and it is an engineering choice rather than a law. Some tokenizers apply NFC or NFKC before tokenizing; many byte-level tokenizers used by large language models apply no normalization at all and preserve the input exactly, leaving the model to learn that variants are related. The tokenizer you build in Chapter 9 preserves input exactly, so that decoding gives back precisely what was encoded, and normalization, where wanted, is a separate, explicit step in the data pipeline (Chapter 18.4). Whatever choice you make, make it once, record it, and apply it identically during training and use. A tokenizer that normalizes during training but not during use sees different token sequences for the same visible text.

---

### 8.5 Characters, words, and bytes as units: the trade-offs

A model needs a fixed vocabulary of units. The three obvious candidates behave very differently. The milestone script trains each kind on the harbor training text and applies it to harbor validation text and to a short multilingual sample:

File: [`code/data/tiny/multilingual.txt`](../../code/data/tiny/multilingual.txt) (one sentence per line, written for this book; the translations of "the keeper lit the lamp" are illustrative, and small wording differences do not affect the measurements)

```text
The keeper lit the lamp.
Le gardien a allumé la lampe.
Der Wärter zündete die Lampe an.
El farero encendió la lámpara.
Смотритель зажёг лампу.
रखवाले ने दीया जलाया।
灯台守がランプをつけた。
灯塔看守人点亮了灯。
أشعل الحارس المصباح.
🌊⚓🐟🚢
for boat in harbor: print(boat.name)
```

File: [`code/scripts/ch08_compare_units.py`](../../code/scripts/ch08_compare_units.py)

```python
"""Chapter 8 milestone: words, characters, and bytes as units, on English and multilingual text.

Each tokenizer is trained (where training applies) on the harbor training text,
then applied to harbor validation text and to a short multilingual sample.

Run from `code/`:  python -m scripts.ch08_compare_units
"""

from __future__ import annotations

import argparse

from llmfp.counting_lm import read_lines, split_into_words
from llmfp.splits import deduplicate, hash_split
from llmfp.tokenizers import ByteTokenizer, CharTokenizer


def word_stats(train: list[str], texts: list[str]) -> tuple[int, int, int]:
    """Vocabulary size, total tokens, and unknown tokens for Chapter 1's word splitting."""
    vocabulary = {word for line in train for word in split_into_words(line, lowercase=False)}
    words = [word for text in texts for word in split_into_words(text, lowercase=False)]
    return len(vocabulary), len(words), sum(word not in vocabulary for word in words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/tiny/harbor_synth.txt")
    parser.add_argument("--multilingual", default="data/tiny/multilingual.txt")
    args = parser.parse_args()

    splits = hash_split(deduplicate(read_lines(args.data)), salt="0")
    train_text = "\n".join(splits.train)
    samples = {"harbor validation": splits.validation, "multilingual": read_lines(args.multilingual)}
    char_tokenizer = CharTokenizer.train(train_text)
    byte_tokenizer = ByteTokenizer()

    for label, lines in samples.items():
        characters = sum(len(line) for line in lines)
        print(f"\n{label}: {len(lines)} lines, {characters} characters")
        print(f"  {'unit':<6} {'vocab size':>10} {'tokens':>7} {'tokens per line':>15} {'unknown':>8} {'all lines round-trip':>21}")
        vocab, tokens, unknown = word_stats(splits.train, lines)
        print(f"  {'word':<6} {vocab:>10} {tokens:>7} {tokens / len(lines):>15.1f} {unknown:>8} {'no (spacing lost)':>21}")
        for name, tokenizer in (("char", char_tokenizer), ("byte", byte_tokenizer)):
            encoded = [tokenizer.encode(line) for line in lines]
            tokens = sum(len(ids) for ids in encoded)
            unknown = sum(ids.count(0) for ids in encoded) if name == "char" else 0
            ok = all(tokenizer.round_trips(line) for line in lines)
            print(f"  {name:<6} {tokenizer.vocab_size:>10} {tokens:>7} {tokens / len(lines):>15.1f} {unknown:>8} {str(ok):>21}")

    print("\nPer line of the multilingual sample (characters / byte tokens):")
    for line in samples["multilingual"]:
        print(f"  {len(line):>3} chars  {len(byte_tokenizer.encode(line)):>3} bytes   {line}")


if __name__ == "__main__":
    main()
```

```bash
python -m scripts.ch08_compare_units
```

Observed output:

```text

harbor validation: 112 lines, 5931 characters
  unit   vocab size  tokens tokens per line  unknown  all lines round-trip
  word           67    1250            11.2        0     no (spacing lost)
  char           30    5931            53.0        0                  True
  byte          256    5931            53.0        0                  True

multilingual: 11 lines, 241 characters
  unit   vocab size  tokens tokens per line  unknown  all lines round-trip
  word           67      68             6.2       53     no (spacing lost)
  char           30     241            21.9       95                 False
  byte          256     374            34.0        0                  True

Per line of the multilingual sample (characters / byte tokens):
   24 chars   24 bytes   The keeper lit the lamp.
   29 chars   30 bytes   Le gardien a allumé la lampe.
   32 chars   34 bytes   Der Wärter zündete die Lampe an.
   30 chars   32 bytes   El farero encendió la lámpara.
   23 chars   43 bytes   Смотритель зажёг лампу.
   21 chars   57 bytes   रखवाले ने दीया जलाया।
   12 chars   36 bytes   灯台守がランプをつけた。
   10 chars   30 bytes   灯塔看守人点亮了灯。
   20 chars   37 bytes   أشعل الحارس المصباح.
    4 chars   15 bytes   🌊⚓🐟🚢
   36 chars   36 bytes   for boat in harbor: print(boat.name)
```

On familiar English text, all three units work; the differences are in sequence length. Words need about 11 tokens per harbor sentence; characters and bytes need about 53. On the multilingual sample, the differences become failures:

| Unit | Vocabulary size | Sequence length | Unknown input | Round trip |
|---|---|---|---|---|
| **Word** | Grows with every new word, name, misspelling, and inflection; real text needs hundreds of thousands | Shortest | Most non-training words are unknown (53 of 68 here) | Loses spacing and exact formatting |
| **Character (code point)** | Small for one language; large if all of Unicode is included (over 100,000 assigned characters) | Long | Any unseen character is unknown (95 of 241 here) | Exact for seen characters only |
| **Byte** | Exactly 256, forever | Longest: 1 byte per ASCII character, 2 to 4 for others | **Nothing is ever unknown** | Always exact |

Read the per-line table at the end of the output: the English sentence and the Python line cost one byte per character, but the Russian line costs almost two bytes per character, the Hindi line nearly three, and the Chinese and Japanese lines three. The emoji line spends 15 bytes on four symbols. **Byte-level units are complete but unequal: the same meaning costs more tokens in some languages than in others.**

Why sequence length matters so much:

- **Context.** A model's context window is measured in tokens (Chapter 11.2). If each token covers less text, the window covers less of the document.
- **Compute and cost.** A transformer's work grows with the number of tokens, and more than proportionally in its attention component (Chapter 13). Commercial APIs bill per token (Chapter 41.7). Text that needs three times as many tokens costs roughly three times as much to process, which is a real fairness concern for languages that tokenize poorly (Chapter 9.9 measures it with a published tokenizer).
- **Learning.** Longer sequences of smaller units make each prediction carry less information, and the model must learn to assemble meaning from more pieces.

None of the three is right. Words are too many and too brittle; characters and bytes make sequences too long. The answer used by almost every modern language model is in between: **subword** units, learned from data, that are whole words for common words and smaller pieces for rare ones, built on top of bytes so nothing is ever unknown. That is byte-pair encoding, the subject of Chapter 9.

---

### 8.6 Tokens, vocabularies, and token IDs

With that background, the terms from Chapter 1 can now be defined precisely.

A **tokenizer** is the component that converts text into a sequence of integers (**encoding**) and converts such a sequence back into text (**decoding**).

A **token** is one unit in that sequence, as defined by a particular tokenizer. It may be a whole word, part of a word, a single character, a single byte, a space followed by a word, or a special marker. "Token" has no meaning independent of the tokenizer: the same text is 1 token for one tokenizer and 10 for another.

The **vocabulary** is the fixed set of all tokens a tokenizer can produce. Its size, `vocab_size`, determines the size of the model's input lookup table (Chapter 10) and output layer (Chapter 5.6): one row per token.

A **token ID** is a token's position in the vocabulary, an integer from 0 to `vocab_size - 1`. As Chapter 7.4 stressed, **token IDs are labels, not measurements**. Token 500 is not "more" than token 250 in any sense. That is why IDs are never fed to a network as quantities, but used to look up a learned list of numbers instead (Chapter 10).

**A tokenizer and a model are a matched pair.** A model's parameters are arranged by token ID. A model trained with one tokenizer and used with another receives meaningless inputs and produces scores for the wrong tokens, without any error. This is why tokenizers are saved alongside model checkpoints (Chapter 7.8), and why loading a pretrained model always means loading its tokenizer too (Chapter 23).

---

### 8.7 A common `Tokenizer` interface; character and byte tokenizers

From here on, every tokenizer in the book follows one interface, so that data pipelines and models can use any of them.

File: [`code/llmfp/tokenizers/base.py`](../../code/llmfp/tokenizers/base.py)

```python
"""The interface every tokenizer in the book follows, and saving/loading (Chapter 8.7)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

FORMAT = "llmfp-tokenizer-v1"
_REGISTRY: dict[str, type["Tokenizer"]] = {}


def register(cls: type["Tokenizer"]) -> type["Tokenizer"]:
    """Class decorator: lets load_tokenizer rebuild this kind of tokenizer from its saved name."""
    _REGISTRY[cls.kind] = cls
    return cls


class Tokenizer(ABC):
    """Text in, integer IDs out, and back again.

    Every subclass sets `kind` (a short name stored in saved files) and implements
    encode, decode, vocab_size, to_dict, and from_dict.
    """

    kind: str = "abstract"

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """How many distinct IDs this tokenizer can produce: valid IDs are 0 .. vocab_size - 1."""

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this tokenizer, as JSON-compatible data."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tokenizer": ...

    def round_trips(self, text: str) -> bool:
        """True if decoding the encoding gives back exactly the original text."""
        return self.decode(self.encode(text)) == text


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "kind": tokenizer.kind, **tokenizer.to_dict()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a saved tokenizer (format={data.get('format')!r})")
    kind = data.pop("kind")
    data.pop("format")
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer kind {kind!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[kind].from_dict(data)
```

- **`Tokenizer`** is an *abstract base class* (ABC): it declares the methods every tokenizer must have, and Python refuses to create an instance of a subclass that has not implemented all of them. The required pieces are `vocab_size`, `encode`, `decode`, and `to_dict`/`from_dict` for saving.
- **`round_trips`** is shared: decode the encoding and compare with the original.
- **`save_tokenizer` and `load_tokenizer`** write and read JSON that records the tokenizer's `kind`. The `@register` decorator adds each tokenizer class to a registry, so `load_tokenizer` can rebuild the right class from a file without the caller knowing which kind it holds. A format tag guards against loading the wrong file, as Chapter 1's checkpoint did.

The byte tokenizer:

File: [`code/llmfp/tokenizers/byte.py`](../../code/llmfp/tokenizers/byte.py)

```python
"""A byte-level tokenizer: every UTF-8 byte value is a token (Chapter 8.7).

There are exactly 256 possible byte values, so the vocabulary is fixed, and any
text in any language encodes without an "unknown" token. The price: most
non-English characters take 2-4 tokens, so sequences get long.
"""

from __future__ import annotations

from typing import Any

from llmfp.tokenizers.base import Tokenizer, register


@register
class ByteTokenizer(Tokenizer):
    kind = "byte"

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        # errors="replace": an incomplete or invalid byte sequence becomes U+FFFD (the
        # replacement character) instead of crashing. This matters when a model generates
        # the first byte of a multi-byte character but not the rest.
        return bytes(ids).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ByteTokenizer":
        return cls()
```

Encoding is UTF-8 encoding; decoding is UTF-8 decoding with replacement for invalid sequences. It needs no training, has exactly 256 IDs, and can encode any text.

The character tokenizer, an improved version of Chapter 7's vocabulary:

File: [`code/llmfp/tokenizers/char.py`](../../code/llmfp/tokenizers/char.py)

```python
"""A character-level tokenizer with an unknown token (Chapter 8.7).

The vocabulary is every character seen in training text, sorted, after one
reserved entry: ID 0 is <unk>, used for any character not seen in training.
Encoding unseen characters therefore never fails, but decoding cannot restore
them: they come back as U+FFFD, the replacement character.
"""

from __future__ import annotations

from typing import Any

from llmfp.tokenizers.base import Tokenizer, register

UNKNOWN = "<unk>"
REPLACEMENT = "�"   # what an unknown token decodes to


@register
class CharTokenizer(Tokenizer):
    kind = "char"

    def __init__(self, characters: list[str]) -> None:
        if len(set(characters)) != len(characters):
            raise ValueError("characters must be unique")
        self.characters = list(characters)
        self.index = {character: i + 1 for i, character in enumerate(self.characters)}  # 0 is <unk>

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.characters) + 1

    @property
    def unknown_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [self.index.get(character, self.unknown_id) for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(REPLACEMENT if i == self.unknown_id else self.characters[i - 1] for i in ids)

    def to_dict(self) -> dict[str, Any]:
        return {"characters": self.characters}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharTokenizer":
        return cls(data["characters"])
```

The difference from Chapter 7: instead of refusing unseen characters, it reserves ID 0 for an **unknown token**, `<unk>`, and encodes any unseen character as 0. Encoding never fails. But information is lost: every unknown character becomes the same ID, so decoding can only return the replacement character, and the round trip fails. The milestone output showed it: "all lines round-trip: False" on the multilingual sample.

The package's `__init__.py` imports the pieces so callers can write `from llmfp.tokenizers import ByteTokenizer`:

File: [`code/llmfp/tokenizers/__init__.py`](../../code/llmfp/tokenizers/__init__.py)

```python
"""Tokenizers: turn text into integer IDs and back (Chapters 8-9).

    Tokenizer        common interface (base.py)
    CharTokenizer    one ID per character seen in training text, plus <unk> (char.py)
    ByteTokenizer    one ID per UTF-8 byte value: always 256 IDs, nothing unknown (byte.py)
    BPETokenizer     byte-level byte-pair encoding (bpe.py, Chapter 9)

    save_tokenizer / load_tokenizer   JSON files that record which kind they hold
"""

from llmfp.tokenizers.base import Tokenizer, load_tokenizer, register, save_tokenizer
from llmfp.tokenizers.byte import ByteTokenizer
from llmfp.tokenizers.char import CharTokenizer

__all__ = ["Tokenizer", "CharTokenizer", "ByteTokenizer", "save_tokenizer", "load_tokenizer", "register"]
```

(The listing mentions `bpe.py`, which Chapter 9 adds.)

---

### 8.8 Round-trip tests: decode(encode(text)) must give back the text

For a lossless tokenizer, decoding the encoding of any text must return exactly that text. This **round-trip property** is the single most important tokenizer test. A tokenizer that silently alters text, by dropping a zero-width joiner, normalizing an accent, or merging two spaces, will train a model on something other than the data and make its output differ from what users typed.

File: [`code/tests/test_tokenizers.py`](../../code/tests/test_tokenizers.py)

```python
"""Tests for llmfp.tokenizers: base, char, and byte tokenizers (Chapter 8)."""

from __future__ import annotations

import random

import pytest

from llmfp.tokenizers import ByteTokenizer, CharTokenizer, load_tokenizer, save_tokenizer
from llmfp.tokenizers.char import REPLACEMENT

TRICKY_TEXTS = [
    "",
    "The keeper lit the lamp.",
    "Café 🐟 at the pier",
    "é vs é",
    "\U0001F468‍\U0001F469‍\U0001F467 family",
    "灯塔看守人点亮了灯。",
    "रखवाले ने दीया जलाया।",
    "tabs\tand\nnewlines\r\n",
    "for boat in harbor: print(boat.name)",
]


def random_unicode_text(generator: random.Random, length: int) -> str:
    """Random code points from all planes, skipping surrogates (which UTF-8 cannot encode)."""
    characters = []
    while len(characters) < length:
        code_point = generator.randrange(0x110000)
        if not 0xD800 <= code_point <= 0xDFFF:
            characters.append(chr(code_point))
    return "".join(characters)


@pytest.mark.parametrize("text", TRICKY_TEXTS)
def test_byte_tokenizer_round_trips_tricky_text(text):
    assert ByteTokenizer().round_trips(text)


def test_byte_tokenizer_round_trips_random_unicode():
    generator = random.Random(0)
    tokenizer = ByteTokenizer()
    for _ in range(200):
        assert tokenizer.round_trips(random_unicode_text(generator, 20))


def test_byte_ids_are_utf8_bytes_and_in_range():
    ids = ByteTokenizer().encode("é🐟")
    assert ids == [195, 169, 240, 159, 144, 159]
    assert all(0 <= i < 256 for i in ids)


def test_byte_decode_of_partial_character_uses_replacement():
    assert ByteTokenizer().decode([67, 240, 159]) == "C�"


def test_char_tokenizer_round_trips_text_it_was_trained_on():
    text = "The keeper lit the lamp.\n"
    tokenizer = CharTokenizer.train(text)
    assert tokenizer.round_trips(text)
    assert tokenizer.vocab_size == len(set(text)) + 1


def test_char_tokenizer_maps_unseen_characters_to_unknown():
    tokenizer = CharTokenizer.train("abc")
    assert tokenizer.encode("abz") == [1, 2, 0]
    assert tokenizer.decode(tokenizer.encode("abz")) == "ab" + REPLACEMENT
    assert not tokenizer.round_trips("abz")


def test_char_ids_are_stable_regardless_of_text_order():
    assert CharTokenizer.train("cab").characters == CharTokenizer.train("abc").characters


@pytest.mark.parametrize("tokenizer", [ByteTokenizer(), CharTokenizer.train("harbor lamp")])
def test_save_and_load(tmp_path, tokenizer):
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    loaded = load_tokenizer(path)
    assert type(loaded) is type(tokenizer)
    assert loaded.encode("harbor lamp!") == tokenizer.encode("harbor lamp!")


def test_load_rejects_other_files(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(path)
```

Three kinds of test cases work together:

- **Hand-picked tricky text** (`TRICKY_TEXTS`): empty strings, accents in both forms, a multi-code-point emoji, several scripts, tabs and Windows line endings, code. Each is a case where a careless tokenizer is known to go wrong.
- **Random text** (`random_unicode_text`): 200 strings of code points drawn from the whole Unicode range, skipping *surrogates* (a reserved range of code points that cannot appear in valid UTF-8). Random inputs find cases nobody thought of. This style, checking a property against many generated inputs, is called *property-based testing*; dedicated libraries exist for it, and this book uses the plain standard-library version.
- **Exact expectations** where the right answer is known: the UTF-8 bytes of `é🐟`, the replacement character for a partial emoji, the unknown-token behavior of the character tokenizer.

Run them:

```bash
pytest tests/test_tokenizers.py -q
```

```text
..................                                                       [100%]
18 passed in 0.03s
```

Chapter 9's BPE tokenizer must pass the same round-trip tests, plus more.

---

### 8.9 Common mistakes, recap, concept checks, exercises, answers, and checkpoint

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| `UnicodeDecodeError` reading a file | File not UTF-8, or read without `encoding="utf-8"` | Specify the encoding; find the file's real encoding and convert it |
| � appears in decoded model output | A byte sequence that is not valid UTF-8, often a character split across tokens | Expected mid-stream; for final output, check the generation stopped mid-character, or that the right tokenizer is used |
| Two visually identical strings compare unequal, or deduplication misses copies | Different normalization forms, zero-width or look-alike characters | Normalize (NFC) consistently in the data pipeline; inspect with `unicodedata.name` |
| Text length limits behave strangely on emoji or accented text | `len()` counts code points, not graphemes or bytes | Decide which you mean; tokenizers count tokens, which differ again |
| A model produces nonsense after loading | Tokenizer and model mismatched | Save and load tokenizer with the model; verify `vocab_size` matches |
| Non-English text "works" but loses characters | A character or word tokenizer mapping unseen input to an unknown token | Use a byte-level tokenizer, and test round trips on multilingual text |
| Token counts for the same content differ widely by language | Byte or subword tokenizers are not equally efficient across scripts | Measure tokens per language for your tokenizer (Ch 9.9); budget context and cost accordingly |

#### Recap

- Unicode assigns every character a **code point**; `len()` counts code points, and a visible symbol can be several of them.
- **UTF-8** writes code points as 1 to 4 bytes; not every byte sequence is valid, and invalid ones decode to the replacement character.
- Identical-looking text can differ underneath. **Normalization** (NFC, NFD, NFKC) makes equivalent forms identical; whether and how to normalize is a recorded engineering choice applied consistently.
- **Words** give short sequences but huge vocabularies and many unknowns; **characters** have unknowns outside their training script; **bytes** never have unknowns but make sequences long, and longer for some languages than others.
- A **token** is whatever a particular tokenizer defines; **token IDs** are labels; a tokenizer and a model must match.
- The book's tokenizers share one interface, save to tagged JSON, and are tested by **round trips** on tricky, random, and exact cases.

#### Concept checks

1. What is the difference between a character as a reader sees it, a code point, and a byte?
2. How many bytes does UTF-8 use for `A`, `é`, `灯`, and `🐟`?
3. Why can decoding a list of byte tokens produce �, and when is that expected?
4. `"café" == "café"` returns `False`. Give two possible reasons.
5. What does NFC normalization do? Why might a tokenizer choose not to normalize?
6. Why is a word-level vocabulary a poor choice for an LLM?
7. Why can a byte-level tokenizer never produce an unknown token?
8. The same sentence takes 24 byte tokens in English and 57 in Hindi. Name two practical consequences.
9. What exactly is a vocabulary, and how does its size affect a model?
10. Why must a tokenizer be saved with the model it was used to train?
11. What does the round-trip property say, and why does the character tokenizer with `<unk>` fail it?
12. Why do the tokenizer tests include randomly generated text as well as hand-picked examples?

#### Exercises

**Exercise 1 (byte prediction).** Without running code, predict the UTF-8 byte count of: `"lamp"`, `"lámpara"`, `"лампа"`, `"ランプ"`, `"🌊⚓"`. Check with `len(text.encode("utf-8"))`. Explain any surprise using section 8.3.

**Exercise 2 (find the invisible difference).** Write a function `explain_difference(a, b)` that, for two strings that compare unequal, prints the first position where they differ and the Unicode names of the code points there. Test it on the composed/decomposed `café` pair, the Latin/Cyrillic `harbor` pair, and `"the lamp"` versus `"the lamp"`.

**Exercise 3 (normalize, then deduplicate).** Count how many of these strings are distinct before and after NFC normalization, and after NFKC: `"café"` (composed), `"café"` (decomposed), `"Café"`, `"café "`, `"ｃａｆé"` (full-width letters). Decide which normalization (if any) you would use for deduplicating harbor log messages, and justify it.

**Exercise 4 (a grapheme-aware length).** Python's standard library has no grapheme splitter. Write a simple approximation that treats a code point as part of the previous grapheme if it is a combining character (`unicodedata.combining(c) != 0`), a zero-width joiner, or follows a zero-width joiner. Compare `len()` with your count on the examples in section 8.2. Where does your approximation fail? (Flags are a good place to look.)

**Exercise 5 (unknown-token rate).** Train a `CharTokenizer` on the English harbor corpus and measure the share of unknown tokens on each line of the multilingual sample. Then train it on the harbor corpus *plus* the multilingual file and measure on a new sentence you write in a language not in the file. What does this say about character vocabularies for open-ended input?

**Exercise 6 (a streaming decoder).** Write a class `StreamingByteDecoder` with a method `add(byte_id) -> str` that returns only the *complete* characters available so far, holding back incomplete bytes, so that feeding the bytes of `"Café 🐟"` one at a time never produces �. Test it, including the case where the stream ends mid-character. (Hint: Python's `codecs.getincrementaldecoder("utf-8")()` does this; write it yourself first, then compare.)

#### Suggested answers and acceptance criteria

**Concept checks**

1. A reader's character (grapheme) is what appears as one symbol; a code point is one Unicode number, and a grapheme can be several; a byte is a unit of storage, and UTF-8 uses 1 to 4 bytes per code point.
2. 1, 2, 3, and 4.
3. A multi-byte character's bytes may arrive in different steps, or a model may generate bytes that are not valid UTF-8. During streaming, an incomplete character is expected; in final output, it signals an invalid sequence.
4. Any two of: composed versus decomposed accents; a look-alike letter from another script; an invisible character such as a zero-width space; a different kind of space.
5. It converts text to composed form, so equivalent sequences become identical code points. A tokenizer may skip it to preserve input exactly, so that decoding returns precisely what was given, leaving normalization to a separate, explicit pipeline step.
6. The vocabulary would need to cover every word, name, inflection, and typo, so it is enormous yet still leaves many inputs unknown, and it loses exact spacing.
7. Every possible text becomes UTF-8 bytes, and all 256 byte values are in the vocabulary.
8. The Hindi text uses more of the context window and costs more compute (and money, with per-token pricing); the model also has to predict more steps per sentence.
9. The fixed set of tokens the tokenizer can produce, each with an ID. Its size sets the number of rows in the model's input lookup table and output layer.
10. The model's parameters are arranged by token ID; with a different tokenizer the same IDs mean different tokens and the model's inputs and outputs become meaningless.
11. Decoding the encoding of any text must return exactly that text. Every unseen character becomes the same unknown ID, so decoding cannot know which character it was.
12. Random inputs find cases nobody anticipated, such as rare characters or unusual combinations; hand-picked cases cover known pitfalls with readable failures.

**Exercise 1.** 4, 8, 10, 9, 7. Acceptance: you check each and explain them: ASCII letters take one byte, `á` and Cyrillic letters two, the Japanese katakana three, emoji four (and the anchor ⚓ three, because it is an older symbol with a smaller code point; this is the usual surprise).

**Exercise 2.** Acceptance: for `café`, the difference is reported at position 3 (`LATIN SMALL LETTER E WITH ACUTE` versus `LATIN SMALL LETTER E`); for `harbor`, at position 1 (`LATIN SMALL LETTER A` versus `CYRILLIC SMALL LETTER A`); for the space, at position 3 (`SPACE` versus `NO-BREAK SPACE`). Your function also handles one string being a prefix of the other.

**Exercise 3.** Before normalization all five differ. NFC merges the two lowercase `café` forms (4 distinct); NFKC additionally maps the full-width letters to plain ones, making that string equal to the composed `café` too (3 distinct: `café`, `Café`, and the one with a trailing space). Acceptance: your counts match and your choice is argued: for deduplicating log messages, NFC plus whitespace trimming is a defensible minimum; NFKC is reasonable when styling never carries meaning; case folding depends on whether case matters in your logs.

**Exercise 4.** Acceptance: your function counts 1 for `"é"`, 1 for the family emoji, and 4 for `"lamp"`, and you identify that it counts the flag as 2, because regional indicators combine in pairs without a joiner. Real grapheme segmentation follows detailed Unicode rules; libraries implement them, and the lesson is that "length" depends on which unit you mean.

**Exercise 5.** The author's measurement (training on the whole harbor corpus): the English sentence has no unknowns; the Russian, Hindi, Japanese, Chinese, Arabic, and emoji lines are almost entirely unknown (for example 20 of 23 Russian characters, all 12 Japanese characters); the French, German, and Spanish lines lose their accented letters *and* letters the harbor corpus happens never to use, such as capital `L`, `D`, `E`, `W` and lowercase `z`; even the Python line loses `(`, `)`, and `:`. Acceptance: your per-line shares show this pattern. After adding the file, the new sentence in another script is again largely unknown. Character vocabularies are closed: any script, symbol, or emoji missing from training data is lost, which is why open-ended systems use bytes underneath.

**Exercise 6.** Acceptance: feeding the 10 bytes of `"Café 🐟"` produces `"C"`, `"a"`, `"f"`, `""`, `"é"`, `" "`, `""`, `""`, `""`, `"🐟"`, and joining them gives the original text; a stream ending mid-character leaves bytes pending, which a `finish()` method reports (as � or an error, by your design choice). Your implementation counts how many continuation bytes a leading byte announces (section 8.3: the first byte signals the length), or uses the incremental decoder; both should pass the same tests.

#### Checkpoint: what you can now do independently

You can now:

- Inspect any string at the level of code points and bytes, and explain its length and size.
- Recognize and fix normalization, whitespace, and look-alike problems in text data.
- Choose between word, character, and byte units with measured trade-offs, and explain why subwords are the usual answer.
- Implement tokenizers against a shared interface, save and load them, and test them with round trips on tricky and random text.

**Next:** Chapter 9 builds a byte-level byte-pair encoding tokenizer, the kind used by GPT-style models, and compares it with a published one.
