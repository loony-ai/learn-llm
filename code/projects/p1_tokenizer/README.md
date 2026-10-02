# Project 1: Build and test a tokenizer

Specification, success criteria, evaluation, debugging exercise, and reviewer checklist:
[Chapter 9.10](../../../book/part-2-text-to-inputs/ch09-byte-pair-encoding.md).

All commands run from `code/` with the `.venv` environment active.

| Step | Command | Output |
|---|---|---|
| 1. Build the fixed training corpus | `python -m scripts.ch09_build_corpus` | `data/tokenizer/corpus.txt` |
| 2. Train the tokenizer | `python -m scripts.ch09_train_bpe` | `data/tokenizer/harbor-bpe-2048.json` + run record |
| 3. Test it | `pytest tests/test_bpe.py tests/test_tokenizers.py` | round trips, determinism, special tokens, save/load |
| 4. Compare with GPT-2 and bytes | `python -m scripts.ch09_compare_tokenizers` | characters per token on held-out text |
| 5. Vocabulary-size study | `python -m solutions.ch09_vocab_sweep` | compression vs vocabulary size |

Implementation: [`llmfp/tokenizers/bpe.py`](../../llmfp/tokenizers/bpe.py). Configuration: [`configs/bpe-cpu.toml`](../../configs/bpe-cpu.toml).
