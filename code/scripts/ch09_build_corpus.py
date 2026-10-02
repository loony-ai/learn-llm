"""Chapter 9: assemble a fixed tokenizer-training corpus from text written for this project.

Sources (all original to this book, so there are no licensing questions):
    data/tiny/harbor_synth.txt              generated harbor sentences
    book/part-1-foundations/*.src.md        Part 1 chapter sources (English prose, Markdown)
    code/llmfp/**/*.py (Chapters 1-8 code)  Python source code

The output is committed to the repository, so every reader trains on identical
text even after the book's own files change.

Run from `code/`:  python -m scripts.ch09_build_corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llmfp.experiment import file_sha256

BOOK = Path("../book/part-1-foundations")
CODE_FILES = [
    "llmfp/counting_lm.py", "llmfp/config.py", "llmfp/devices.py", "llmfp/experiment.py",
    "llmfp/splits.py", "llmfp/counting_eval.py", "llmfp/nn_basics.py", "llmfp/training_basics.py",
    "llmfp/toy_data.py", "llmfp/char_model.py", "llmfp/tokenizers/base.py",
    "llmfp/tokenizers/char.py", "llmfp/tokenizers/byte.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/tokenizer/corpus.txt")
    args = parser.parse_args()

    parts = {"harbor": Path("data/tiny/harbor_synth.txt").read_text(encoding="utf-8")}
    parts["book"] = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(BOOK.glob("ch0*.src.md")))
    parts["code"] = "\n\n".join(Path(f).read_text(encoding="utf-8") for f in CODE_FILES)
    corpus = "\n\n".join(parts[name] for name in ("harbor", "book", "code"))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    for name, text in parts.items():
        print(f"{name:<7} {len(text):>8,} characters")
    print(f"total   {len(corpus):>8,} characters -> {output} (sha256 {file_sha256(output)[:16]}...)")


if __name__ == "__main__":
    main()
