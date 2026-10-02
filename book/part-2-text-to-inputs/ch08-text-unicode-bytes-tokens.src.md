## Chapter 8: Text, Unicode, Bytes, and the Need for Tokens

[Back to index](../../README.md) · Previous: [Chapter 7](../part-1-foundations/ch07-project-0-char-model.md) · Next: [Chapter 9](ch09-byte-pair-encoding.md)

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
@@FILE code/examples/ch08/unicode_basics.py@@
```

Observed output:

```text
@@RUN python examples/ch08/unicode_basics.py@@
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
@@FILE code/examples/ch08/utf8_bytes.py@@
```

Observed output:

```text
@@RUN python examples/ch08/utf8_bytes.py@@
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
@@FILE code/examples/ch08/normalization.py@@
```

Observed output:

```text
@@RUN python examples/ch08/normalization.py@@
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
@@FILE code/data/tiny/multilingual.txt@@
```

File: [`code/scripts/ch08_compare_units.py`](../../code/scripts/ch08_compare_units.py)

```python
@@FILE code/scripts/ch08_compare_units.py@@
```

```bash
python -m scripts.ch08_compare_units
```

Observed output:

```text
@@RUN python -m scripts.ch08_compare_units@@
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
@@FILE code/llmfp/tokenizers/base.py@@
```

- **`Tokenizer`** is an *abstract base class* (ABC): it declares the methods every tokenizer must have, and Python refuses to create an instance of a subclass that has not implemented all of them. The required pieces are `vocab_size`, `encode`, `decode`, and `to_dict`/`from_dict` for saving.
- **`round_trips`** is shared: decode the encoding and compare with the original.
- **`save_tokenizer` and `load_tokenizer`** write and read JSON that records the tokenizer's `kind`. The `@register` decorator adds each tokenizer class to a registry, so `load_tokenizer` can rebuild the right class from a file without the caller knowing which kind it holds. A format tag guards against loading the wrong file, as Chapter 1's checkpoint did.

The byte tokenizer:

File: [`code/llmfp/tokenizers/byte.py`](../../code/llmfp/tokenizers/byte.py)

```python
@@FILE code/llmfp/tokenizers/byte.py@@
```

Encoding is UTF-8 encoding; decoding is UTF-8 decoding with replacement for invalid sequences. It needs no training, has exactly 256 IDs, and can encode any text.

The character tokenizer, an improved version of Chapter 7's vocabulary:

File: [`code/llmfp/tokenizers/char.py`](../../code/llmfp/tokenizers/char.py)

```python
@@FILE code/llmfp/tokenizers/char.py@@
```

The difference from Chapter 7: instead of refusing unseen characters, it reserves ID 0 for an **unknown token**, `<unk>`, and encodes any unseen character as 0. Encoding never fails. But information is lost: every unknown character becomes the same ID, so decoding can only return the replacement character, and the round trip fails. The milestone output showed it: "all lines round-trip: False" on the multilingual sample.

The package's `__init__.py` imports the pieces so callers can write `from llmfp.tokenizers import ByteTokenizer`:

File: [`code/llmfp/tokenizers/__init__.py`](../../code/llmfp/tokenizers/__init__.py)

```python
@@FILE code/llmfp/tokenizers/__init__.py@@
```

(The listing mentions `bpe.py`, which Chapter 9 adds.)

---

### 8.8 Round-trip tests: decode(encode(text)) must give back the text

For a lossless tokenizer, decoding the encoding of any text must return exactly that text. This **round-trip property** is the single most important tokenizer test. A tokenizer that silently alters text, by dropping a zero-width joiner, normalizing an accent, or merging two spaces, will train a model on something other than the data and make its output differ from what users typed.

File: [`code/tests/test_tokenizers.py`](../../code/tests/test_tokenizers.py)

```python
@@FILE code/tests/test_tokenizers.py@@
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
@@RUN pytest tests/test_tokenizers.py -q -p no:cacheprovider@@
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

**Next:** [Chapter 9](ch09-byte-pair-encoding.md) builds a byte-level byte-pair encoding tokenizer, the kind used by GPT-style models, and compares it with a published one.
