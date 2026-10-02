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
