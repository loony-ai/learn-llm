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
