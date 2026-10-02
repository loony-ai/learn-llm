## Editorial Ledger

[Back to index](../../README.md)

The ledger is the book's source of truth for what has been delivered, what depends on what, and what must be revisited. Update it with every chapter.

**Last updated:** 2026-10-02, after Chapter 1.

### 1. Completed chapters and outstanding sections

| Chapter | Status | Notes |
|---|---|---|
| Planning materials | Done | Learning journey, contents, prerequisite map, repository structure, capstone map, hardware paths, coverage audit, this ledger |
| 1 What a language model is and what it predicts | Done | All 15 sections, exercises, and answers |
| 2–45 | Not started | Next: Chapter 2 |
| Appendices A–G | Not started | The glossary will be seeded from the ledger's concept table |

### 2. Concepts introduced

Only concepts that have been *delivered* are listed. Planned locations are in the [prerequisite map](prerequisite-map.md).

| Concept | First explained | Depth so far | Revisit in |
|---|---|---|---|
| Artificial intelligence (AI) | 1.2 | Definition, scope | — |
| Machine learning (ML) | 1.2, 1.3 | Definition; rules vs learned behavior | Ch 4–6 |
| Deep learning | 1.2 | Definition only (neural networks named, not explained) | Ch 5 |
| Natural language processing (NLP) | 1.2 | Definition | — |
| Language model; large language model (LLM) | 1.2, 1.6 | Definition; what "large" refers to | Ch 12, 16 |
| Model | 1.4 | Architecture + parameters | Ch 5 |
| Architecture | 1.4 | Counting-model example | Ch 12 |
| Parameter / weight | 1.4 | Counts as parameters; neural weights described in words | Ch 5.3 |
| Hyperparameter | 1.4 | `context_size`, `lowercase` | Ch 4.7, 19 |
| Checkpoint | 1.4 | JSON file of config + counts | Ch 7.8, 19.9 |
| Training / inference | 1.5 | Counting vs lookup | Ch 6 |
| Pretraining / fine-tuning / adaptation | 1.5 | Life-cycle overview | Ch 25 |
| Token (informal) | 1.6 | "Word or punctuation mark" for now | Ch 8.6 (formal) |
| Context, context size | 1.6 | Last N tokens | Ch 11.2 |
| Next-token candidates with scores | 1.6 | Counts as scores | Ch 5.7 (logits) |
| Autoregressive generation | 1.7 | Loop diagram + code | Ch 17 |
| Greedy vs sampled choice | 1.7, 1.10 | Behavioral | Ch 21 |
| Special tokens `<start>`, `<end>` | 1.8 | Why they exist | Ch 9.6 |
| Stop conditions | 1.8, 1.10 | Three stop reasons | Ch 21.6 |
| Vocabulary (informal) | 1.10 | Set of words the model can output | Ch 8.6 |
| Random seed (informal) | 1.10 | Same seed, same samples | Ch 4.6 |
| Memorization vs generalization (informal) | 1.11 | Observed via context size | Ch 4.5 |
| Data sparsity / unseen contexts | 1.11 | Observed: no prediction | Ch 5.1 |
| Hallucination (informal) | 1.12 | Fluent but false output, observed | Ch 21.9, 38.3 |
| Anthropomorphic language and its limits | 1.12 | Guidance on "knows/thinks/remembers" | Throughout |

### 3. Prerequisites and unresolved dependencies

| Item | Status |
|---|---|
| Chapter 1 requires only: Python 3.10+ syntax, running a script from a terminal | Satisfied (stated in 1's prerequisites) |
| Forward references in 1.13 (tokens, embeddings, neural networks, attention) | Named only as signposts with chapter links; not relied on |
| `unittest` used in Ch 1 tests before pytest is taught | Explained in 1.10; pytest arrives in 2.10 |
| None unresolved | — |

### 4. Repository files and interfaces

| File | Created | Status | Public interface |
|---|---|---|---|
| `code/llmfp/__init__.py` | Ch 1 | Tested | `__version__ = "0.1.0"` |
| `code/llmfp/counting_lm.py` | Ch 1 | Tested | `START`, `END`, `CountingModelConfig`, `GenerationResult`, `split_into_words`, `join_words`, `rank_followers`, `CountingLanguageModel` (`train`, `context_for`, `followers_for`, `next_word_candidates`, `generate`, `num_parameters`, `num_contexts`, `vocabulary`, `save`, `load`), `read_lines` |
| `code/scripts/__init__.py` | Ch 1 | — | Makes `scripts` importable as a package |
| `code/scripts/ch01_counting_demo.py` | Ch 1 | Executed | CLI: `--data --context-size --prompt --samples --max-new-words --seed --checkpoint` |
| `code/tests/test_counting_lm.py` | Ch 1 | 18 tests pass | — |
| `code/tests/test_ch01_solutions.py` | Ch 1 | 4 tests pass | — |
| `code/solutions/ch01_backoff.py` | Ch 1 | Tested, executed | `BackoffLanguageModel(config)`; `.last_context_size_used` |
| `code/solutions/ch01_memorization.py` | Ch 1 | Executed | CLI: `--data --samples --max-context-size --seed` |
| `code/README.md` | Ch 1 | — | How to run code |
| `tools/build_book.py` | Ch 1 | Executed | Expands `@@FILE@@` and `@@RUN@@` markers in `*.src.md` |
| `code/data/tiny/harbor.txt` | Ch 1 | — | 40 original sentences, written for this book (no third-party license) |

### 5. Dependency versions

| Dependency | Needed from | Version used in testing | Latest on PyPI (checked 2026-10-02) | Pin status |
|---|---|---|---|---|
| Python | Ch 1 | 3.14.4 | — | Book requires 3.12+ (decided; see note) |
| NumPy | Ch 3 | 2.5.1 (installed) | 2.5.3 (requires Python 3.12+) | To pin in Ch 2 |
| PyTorch (`torch`) | Ch 3 | 2.13.0+cpu (installed) | 2.14.1 | To pin in Ch 2 after testing |
| pytest | Ch 2 (Ch 1 tests also pass under it) | 9.1.1 | 9.1.1 | To pin in Ch 2 |
| transformers | Ch 23 | — | 5.18.0 | Verify APIs in Ch 23 |
| tokenizers | Ch 9.8 | — | 0.23.2 | Verify in Ch 9 |
| datasets | Ch 18 | — | 5.0.1 | Verify in Ch 18 |
| safetensors | Ch 22 | — | 0.8.0 | Verify in Ch 22 |
| huggingface-hub | Ch 22 | — | 2.1.1 | Verify in Ch 22 |
| accelerate | Ch 23 | — | 1.15.0 | Verify in Ch 23 |
| peft | Ch 28 | — | 0.21.2 | Verify in Ch 28 |
| bitsandbytes | Ch 28.5 (GPU path) | — | 0.50.2 | Verify backend support in Ch 28 |

Note: Chapter 1 code was written to need only Python 3.10+ features, but it was **executed only on Python 3.14.4**; no other version was available on the test machine. The book as a whole sets 3.12 as its minimum because NumPy's current release requires it.

### 6. Implemented and tested milestones

| Milestone | Command (run from `code/`) | Result | Date |
|---|---|---|---|
| Ch 1 tests (unittest) | `python3 -m unittest discover -s tests -v` | 22 tests, all OK (18 model + 4 solutions) | 2026-10-02 |
| Ch 1 tests (pytest) | `python3 -m pytest -q tests` | 22 passed | 2026-10-02 |
| Ch 1 solutions | `python3 -m solutions.ch01_backoff`, `python3 -m solutions.ch01_memorization` | Output inserted into the chapter by the build tool | 2026-10-02 |
| Ch 1 standalone snippets (1.3, Exercise 2) | Executed by extracting them from the built chapter | Printed results match the comments in the text | 2026-10-02 |
| Ch 1 build | `python3 tools/build_book.py ch01` | All markers expanded | 2026-10-02 |
| Ch 1 demo, default | `python3 -m scripts.ch01_counting_demo` | Output pasted in 1.10 as observed | 2026-10-02 |
| Ch 1 demo, context sizes 1 and 3, unseen prompt, Exercise 1 and 3 prompts | See 1.11, 1.15 | Output pasted (abridged) as observed | 2026-10-02 |
| Ch 1 Exercise 7 (character-level) | Throwaway experiment in scratch space, not in repository | Numbers quoted in answers, labeled as the author's run | 2026-10-02 |

### 7. Teaching simplifications to revisit

| Simplification | Introduced | Where it is corrected or deepened |
|---|---|---|
| Tokens are treated as words and punctuation marks | 1.6, 1.8 | Ch 8–9: subword and byte-level tokens |
| Lowercasing all text | 1.8 | Ch 8.4: why modern tokenizers keep case |
| "Scores" are raw counts | 1.6 | Ch 5.7: logits; Ch 6.2: loss |
| Training = counting; no optimization | 1.5, 1.8 | Ch 6: training as repeated measured adjustment |
| Each line is one sentence; START padding | 1.8 | Ch 11.7: document boundaries and packing |
| Checkpoint = config + parameters only | 1.8 | Ch 19.9: optimizer and random state for resuming training |
| Context is a fixed number of previous words; exact match required | 1.11 | Ch 5, 7: generalization; Ch 13: attention over a long window |
| Random sampling in proportion to counts | 1.7, 1.8 | Ch 21: temperature, top-k, top-p |
| "Copy of a training sentence" checked by exact string match | 1.10 | Ch 18.5, 37.4: near-duplicates and contamination |

### 8. Open questions and decisions

| Decision | Status |
|---|---|
| Default pretraining dataset | Candidate: TinyStories (`roneneldan/TinyStories`, Hub card lists CDLA-Sharing-1.0, checked 2026-10-02). Final choice and license reading in Ch 18 |
| Classification dataset (Ch 26) | Open. The Hub mirror `ucirvine/sms_spam` lists its license as "unknown" (checked 2026-10-02); it will not be used unless the license is confirmed. May write an original harbor dataset instead |
| Pretrained weights for Ch 22 | GPT-2 (`openai-community/gpt2`, Hub lists MIT, checked 2026-10-02) |
| Small instruction models for Parts 5–7 | Candidates (Hub lists Apache-2.0, checked 2026-10-02): `HuggingFaceTB/SmolLM2-135M-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`. Chosen by measurement in Ch 24 |
| Embedding model for Ch 32 | Candidate: `sentence-transformers/all-MiniLM-L6-v2` (Hub lists Apache-2.0, checked 2026-10-02) |
| Serving framework for Ch 40 | Open; decide and verify in Ch 40 |
| Config file format | TOML for hand-written configs (read with the standard library's `tomllib`), JSON for machine-written records |
