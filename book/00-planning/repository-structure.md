## Repository Structure

[Back to index](../../README.md)

The book and its code live in one repository. Markdown is under `book/`; runnable code is under `code/`. Paths marked *(planned)* do not exist yet; the [editorial ledger](editorial-ledger.md) records when each is created and tested.

### Top level

```text
learn-llm/
├── README.md                      # Book title page and index
├── book/
│   ├── 00-planning/               # Journey, contents, maps, ledger
│   ├── part-1-foundations/        # Chapters 1-7
│   ├── part-2-text-to-inputs/     # Chapters 8-11          (planned)
│   ├── part-3-transformer/        # Chapters 12-17         (planned)
│   ├── part-4-training/           # Chapters 18-21         (planned)
│   ├── part-5-pretrained/         # Chapters 22-24         (planned)
│   ├── part-6-adaptation/         # Chapters 25-29         (planned)
│   ├── part-7-applications/       # Chapters 30-36         (planned)
│   ├── part-8-production/         # Chapters 37-42         (planned)
│   ├── part-9-advanced/           # Chapters 43-45         (planned)
│   └── appendices/                # Appendices A-G         (planned)
├── tools/
│   ├── build_book.py              # Expands chapter sources into the chapters you read
│   └── audit_chapters.py          # Checks built chapters: math symbols, filler words, links, markers
└── code/                          # Companion code (run everything from here)
```

#### Chapter sources and generated chapters

Each chapter has a source file, `chNN-*.src.md`, next to the chapter file, `chNN-*.md`. The source contains markers on lines of their own: `@@FILE path@@` inserts a file from the repository verbatim, and `@@RUN command@@` runs a command in `code/` and inserts its output. `python3 tools/build_book.py` regenerates the chapter files. This guarantees that code listings match the tested files and that every block labeled *Observed output* was produced by the code as it currently stands. Edit the `.src.md` file, never the generated `.md` file.

### The `code/` directory

All commands in the book run from inside `code/`. Scripts are run as modules (`python -m scripts.<name>`), which puts `code/` on Python's import path so `import llmfp` works even before Chapter 2 installs the package.

```text
code/
├── pyproject.toml                 # Ch 2  package metadata + pinned dependencies
├── requirements/
│   └── linux-cpu-lock.txt         # Ch 2  exact versions of every package (Linux, CPU)
├── .venv/                         # Ch 2  virtual environment (not committed)
├── llmfp/                         # The book's package ("LLM From First Principles")
│   ├── __init__.py                # Ch 1
│   ├── counting_lm.py             # Ch 1  counting next-word model
│   ├── config.py                  # Ch 2  TOML configs, --set overrides, validation, JSON records
│   ├── devices.py                 # Ch 3  pick_device(), add_device_argument(), tensor_bytes(), format_bytes()
│   ├── experiment.py              # Ch 4  run directories, seeds, experiment records
│   ├── splits.py                  # Ch 4  deterministic train/val/test splitting
│   ├── nn_basics.py               # Ch 5  small modules used to teach layers
│   ├── training_basics.py         # Ch 6  minimal training loop
│   ├── char_model.py              # Ch 7  next-character network
│   ├── tokenizers/                # Ch 8-9
│   │   ├── base.py                #   Tokenizer protocol: encode, decode, vocab_size, save, load
│   │   ├── char.py, byte.py       #   Ch 8
│   │   └── bpe.py                 #   Ch 9
│   ├── data/                      # Ch 11, 18
│   │   ├── windows.py             #   sliding-window next-token dataset
│   │   ├── collate.py             #   padding, masks, packing
│   │   └── pipeline.py            #   download → clean → dedup → split → tokenize → shard
│   ├── model/                     # Ch 10, 12-17
│   │   ├── config.py              #   GPTConfig
│   │   ├── embeddings.py          #   token + learned position embeddings
│   │   ├── attention.py           #   single-head, causal, multi-head attention
│   │   ├── layers.py              #   feed-forward, LayerNorm/RMSNorm
│   │   ├── positions.py           #   rotary positional embeddings
│   │   ├── block.py               #   TransformerBlock
│   │   ├── gpt.py                 #   GPT model
│   │   └── kv_cache.py            #   key-value cache
│   ├── generation/                # Ch 17, 21
│   │   ├── generate.py            #   generation loop
│   │   └── decoding.py            #   greedy, temperature, top-k, top-p, penalties
│   ├── training/                  # Ch 19-20
│   │   ├── trainer.py, schedules.py, checkpoint.py, metrics.py
│   ├── pretrained/                # Ch 22-24
│   │   ├── gpt2_loader.py, hf_models.py, select.py
│   ├── adapt/                     # Ch 26-29
│   │   ├── classification.py, instruction.py, lora.py, dpo.py
│   ├── apps/                      # Ch 30-36
│   │   ├── llm_client.py, prompts.py, structured.py, embeddings.py,
│   │   ├── chunking.py, vector_store.py, rerank.py, rag.py,
│   │   └── tools.py, agent.py, memory.py, security.py
│   ├── eval/                      # Ch 37-38
│   │   ├── datasets.py, metrics.py, judges.py, regression.py
│   └── serving/                   # Ch 40-41
│       ├── app.py, limits.py, monitoring.py
├── scripts/                       # One runnable entry point per milestone: chNN_<what>.py
│   ├── __init__.py
│   ├── ch01_counting_demo.py      # Ch 1
│   ├── ch02_check_env.py          # Ch 2  verify interpreter, venv, pinned versions
│   ├── ch02_train_counting.py     # Ch 2  config-driven training of the counting model
│   └── ch03_tensor_tour.py        # Ch 3  devices, memory estimates, matmul timing
├── tests/                         # pytest-compatible tests, mirroring llmfp/
│   ├── test_counting_lm.py        # Ch 1
│   ├── test_ch01_solutions.py     # Ch 1 exercise solutions
│   ├── test_config.py             # Ch 2
│   ├── test_ch02_solutions.py     # Ch 2 exercise solutions
│   ├── test_devices.py            # Ch 3
│   └── test_ch03_solutions.py     # Ch 3 exercise solutions
├── solutions/                     # Suggested exercise solutions: chNN_<exercise>.py
│   ├── ch01_backoff.py            # Ch 1, Exercise 5
│   ├── ch01_memorization.py       # Ch 1, Exercise 6
│   ├── ch02_iter_sentences.py     # Ch 2, Exercise 4
│   └── ch03_pad_and_stack.py      # Ch 3, Exercise 5
├── examples/                      # Small standalone teaching programs: examples/chNN/*.py
│   ├── ch02/                      # Ch 2  collections, functions, classes, generators, files, pytest failure demo
│   └── ch03/                      # Ch 3  arrays, shapes, dtypes, indexing, reshaping, broadcasting, reductions, batching
├── configs/                       # TOML experiment configs (<purpose>-cpu.toml, <purpose>-gpu.toml)
│   └── counting-cpu.toml          # Ch 2
├── data/
│   ├── tiny/harbor.txt            # Ch 1: 40 original sentences (written for this book)
│   ├── handbook/                  # Ch 33: original Harbor Handbook (planned)
│   └── downloads/                 # Ch 18: fetched datasets (not committed)
├── projects/                      # Capstones: README, scripts, eval sets, checklists
│   ├── p1_tokenizer/ ... p7_serve/
└── runs/                          # Outputs: checkpoints, logs, records (not committed)
```

### Stable interfaces

These signatures are promises. A later chapter may *add* parameters with defaults, but must not break existing callers without an explicit correction note in the chapter and the ledger.

| Interface | Introduced | Signature (summary) |
|---|---|---|
| `CountingModelConfig` | Ch 1 | `CountingModelConfig(context_size=2, lowercase=True)` |
| `CountingLanguageModel` | Ch 1 | `.train(lines) -> int`, `.followers_for(words) -> Counter or None`, `.next_word_candidates(prompt, top=5)`, `.generate(prompt, max_new_words=20, rng=None, greedy=False) -> GenerationResult`, `.save(path)`, `.load(path)` |
| `load_config` | Ch 2 | `load_config(cls, path=None, overrides=None) -> cls`; also `from_dict`, `apply_overrides`, `parse_value`, `to_dict`, `save_json`, `load_json`, `ConfigError` |
| Script config convention | Ch 2 | `--config <file.toml>`, repeatable `--set key=value`, `--log-level`; resolved config saved as JSON beside outputs |
| Device selection | Ch 3 | `pick_device(preference="auto") -> torch.device`; every PyTorch script takes `--device auto\|cpu\|cuda\|mps` via `add_device_argument` |
| `Tokenizer` protocol | Ch 8 | `.encode(text) -> list[int]`, `.decode(ids) -> str`, `.vocab_size`, `.save(path)`, `.load(path)` |
| `GPTConfig` | Ch 12 | dataclass: `vocab_size, context_length, d_model, n_heads, n_layers, dropout, ...` |
| `GPT.forward` | Ch 16 | `(token_ids[B, T], attention_mask=None, kv_cache=None) -> logits[B, T, vocab_size]` |
| `generate` | Ch 17 | `generate(model, token_ids, max_new_tokens, decoding=DecodingConfig(), stop_ids=())` |
| `LLMClient` protocol | Ch 30 | `.complete(messages, **options) -> Completion`, `.stream(...)` |

### Naming conventions

- Scripts: `scripts/chNN_<verb>_<thing>.py`, runnable with `python -m scripts.chNN_...`.
- Configs: `configs/<purpose>-<cpu|gpu>.toml`.
- Runs: `runs/<chapter-or-project>/<run-name>/` containing `config.json`, `record.json`, `checkpoint*`, `log.txt`.
- Tests: `tests/test_<module>.py`, written with `unittest` in Chapter 1 (no installation required) and with pytest from Chapter 2.
- Examples: `examples/chNN/<topic>.py`, run with `python examples/chNN/<topic>.py` from `code/`. A file named `test_*.py` in `examples/` is a deliberate demonstration and is not collected by a plain `pytest` run.
- Environment: all commands from Chapter 2 on assume the `code/.venv` environment is active.
