## Editorial Ledger

[Back to index](../../README.md)

The ledger is the book's source of truth for what has been delivered, what depends on what, and what must be revisited. Update it with every chapter.

**Last updated:** 2026-10-02, after Chapter 4.

### 1. Completed chapters and outstanding sections

| Chapter | Status | Notes |
|---|---|---|
| Planning materials | Done | Learning journey, contents, prerequisite map, repository structure, capstone map, hardware paths, coverage audit, this ledger |
| 1 What a language model is and what it predicts | Done | All 15 sections, exercises, and answers |
| 2 Python foundations and your working environment | Done | Sections 2.1–2.13 (2.13 added for recap/exercises), 6 exercises with answers |
| 3 Tensors | Done | Sections 3.1–3.14 (3.14 added for recap/exercises), 6 exercises with answers |
| 4 Data, experiments, reproducibility | Done | Sections 4.1–4.10 (4.9 retitled; 4.10 added), 6 exercises with answers |
| 5–45 | Not started | Next: Chapter 5 |
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
| Interpreter; `sys.executable` | 2.2 | Several interpreters per machine | 2.12 |
| Package/distribution, PyPI, pip, wheel | 2.2 | Wheel filename anatomy; install vs import name | Ch 23 (Hub downloads) |
| site-packages, import path (`sys.path`) | 2.2 | Explains Ch 1's `-m` trick | — |
| Virtual environment | 2.2, 2.3 | Create, activate, rebuild | Appendix B |
| Direct/transitive dependency, pinning, lock file | 2.2, 2.3 | Exact pins + Linux CPU lock file | Ch 41.5 (versioning) |
| PyTorch build variants (CPU, CUDA, MPS); local version label `+cpu` | 2.3 | Install commands per platform | Ch 3.11 (devices), Ch 19.10 |
| Editable install; extras (`.[dev]`) | 2.3 | — | — |
| `pyproject.toml` | 2.3 | Every section explained | — |
| Hash / hashable (informal) | 2.4 | Why tuples can be dict keys | Ch 18.5 (dedup hashing) |
| Frozen dataclass, `default_factory`, `replace` | 2.4 | Config objects | Ch 12.7 (`GPTConfig`) |
| Type hints (not enforced) | 2.5 | Demonstrated | 2.9 (runtime validation) |
| Keyword-only parameters | 2.5 | Book-wide convention for options | — |
| Mutable default trap | 2.5 | Demonstrated | — |
| Input/target shift (preview) | 2.5, 2.10 | Example + failing test | Ch 11.3 (formal) |
| Properties, class methods, dunder methods, inheritance | 2.6 | Vocabulary example | Ch 5.5, Ch 11.8 |
| `__call__` → `forward` pattern (imitation of PyTorch) | 2.6 | Labeled teaching imitation | Ch 5.5 (real `nn.Module`) |
| Iterator vs iterable; generator; laziness | 2.7 | Demonstrated | Ch 11.8, Ch 18 |
| One-shot iterator trap | 2.7 | Linked to Ch 1 backoff `list(lines)` | — |
| Batching a stream (preview) | 2.7 | `batched` generator | Ch 11.5 |
| Text vs bytes; encoding; UTF-8 (introductory) | 2.8 | Byte counts, wrong-encoding demo | Ch 8.2–8.3 (full) |
| JSON vs TOML; `tomllib` read-only | 2.8 | Convention: TOML in, JSON out | — |
| Configuration layering, overrides, validation | 2.9 | `llmfp/config.py` | Ch 4.7, Ch 19 |
| Logging levels; print vs log | 2.9 | Used in `ch02_train_counting` | Ch 41.2–41.3 |
| pytest: discovery, assert introspection, fixtures, parametrize, raises, selection | 2.10 | — | Ch 16.8 (model fixtures) |
| Exit status | 2.12 | 0 = success convention | Ch 38.5 (CI) |
| Array / tensor; element | 3.2 | Why arrays beat lists (observed 17x on test machine) | — |
| Shape, dimension, axis; scalar/vector/matrix | 3.3 | Axis meaning is a convention to document | App. C, Ch 16.7 |
| dtype; precision; range; int64/bool/float32/float16/bfloat16 | 3.4 | Observed precision/overflow | Ch 19.10, Ch 24.4 |
| Indexing, slicing, boolean masks, table lookup by IDs | 3.5 | `table[ids]` previews embeddings | Ch 10.3 |
| View vs copy; `clone` | 3.5 | — | — |
| reshape / view / transpose / contiguity / unsqueeze / squeeze | 3.6 | Head split previewed | Ch 14.3 |
| Broadcasting (rule in words) and the silent `(N,)` vs `(N,1)` bug | 3.7 | — | Ch 13, 16.8 |
| Reductions; argmax; topk; keepdim | 3.8 | argmax = greedy choice | Ch 5.7, 17, 21.5 |
| Batch, batch dimension; stack vs cat; padding (preview) | 3.9 | Exercise 5 pad_and_stack | Ch 11.5–11.6 |
| NumPy↔PyTorch; float64 default trap | 3.10 | — | — |
| CPU, GPU, RAM, VRAM, MPS, device, `.to(device)` | 3.11 | VRAM as binding limit | Ch 19, 24, 39 |
| Memory estimate: elements times bytes per element | 3.12 | Lower bound only | Ch 16.6, 19, 39 |
| Matrix multiplication (named only, used as benchmark) | 3.12 | Not explained yet | Ch 5.3 (by behavior) |
| Shape-debugging habits | 3.13 | 7 habits + error table | Ch 16.9, App. D |
| Dataset, example, input, label; self-supervised (named) | 4.2 | — | Ch 11 |
| Training/validation/test splits; generalization | 4.3 | Test used once | Ch 18, 37 |
| Hash function, SHA-256, stable hash split | 4.3 | Duplicates grouped; stable as data grows | Ch 18.5 |
| Leakage (duplicates, groups, preprocessing, time, peeking); contamination (named) | 4.4 | Measured: 66.7% vs 45.6% val acc at ctx 10 | Ch 18.6, 20.6, 37.4 |
| Accuracy, coverage (as shares of positions) | 4.4 | Counting model only | Ch 6 (loss), Ch 19.3 |
| Overfitting, underfitting | 4.5 | Measured via context size | Ch 19.4 |
| Irreducible uncertainty (informal) | 4.5 | Train acc stays ~76% | Ch 19.3 |
| Pseudo-random generator, seed limits | 4.6 | Extra draw shifts sequence | Ch 5–7 |
| PYTHONHASHSEED / set order nondeterminism | 4.6 | Demonstrated | — |
| GPU nondeterminism; deterministic algorithms (named) | 4.6 | Not executed | Ch 19 |
| Experiment record; environment capture; data fingerprint | 4.8 | `start_run`/`finish_run` | Ch 19, 41.5 |
| Judging differences by repeated measurement with irrelevant variation | 4.9 | Salt spread | Ch 37.7 |

### 3. Prerequisites and unresolved dependencies

| Item | Status |
|---|---|
| Chapter 1 requires only: Python 3.10+ syntax, running a script from a terminal | Satisfied (stated in 1's prerequisites) |
| Forward references in 1.13 (tokens, embeddings, neural networks, attention) | Named only as signposts with chapter links; not relied on |
| `unittest` used in Ch 1 tests before pytest is taught | Explained in 1.10; pytest arrives in 2.10 (resolved) |
| Ch 2 forward references: Dataset `__getitem__` (Ch 11), `nn.Module` (Ch 5), CI (Ch 38), Appendix B | Named as signposts only; not relied on |
| None unresolved | — |

### 4. Repository files and interfaces

| File | Created | Status | Public interface |
|---|---|---|---|
| `code/llmfp/__init__.py` | Ch 1, updated Ch 2 | Tested | `__version__ = "0.2.0"` (Ch 1 listing shows its own 0.1.0 version inline) |
| `code/llmfp/config.py` | Ch 2 | Tested (21 tests) | `ConfigError`, `load_toml`, `parse_value`, `apply_overrides`, `from_dict`, `load_config(cls, path=None, overrides=None)`, `to_dict`, `save_json`, `load_json` |
| `code/pyproject.toml` | Ch 2 | Installed (editable) | Package `llmfp` 0.2.0; pins numpy, torch; extra `dev` pins pytest |
| `code/requirements/linux-cpu-lock.txt` | Ch 2 | Verified in a second fresh venv | Full `pip freeze` of tested env |
| `code/configs/counting-cpu.toml` | Ch 2 | Tested (matches dataclass defaults) | — |
| `code/scripts/ch02_check_env.py` | Ch 2 | Executed inside and outside venv | Exit 0/1; searches site-packages only |
| `code/scripts/ch02_train_counting.py` | Ch 2 | Executed | `CountingRunConfig`; CLI `--config`, `--set`, `--log-level` |
| `code/examples/ch02/*.py` | Ch 2 | Executed | Teaching programs; `test_failure_demo.py` fails deliberately |
| `code/solutions/ch02_iter_sentences.py` | Ch 2 | Tested, executed | `iter_sentences(paths)` |
| `code/tests/test_config.py` | Ch 2 | 21 tests pass | — |
| `code/tests/test_ch02_solutions.py` | Ch 2 | 8 tests pass | — |
| `code/llmfp/devices.py` | Ch 3 | Tested (CPU paths) | `DEVICE_CHOICES`, `available_devices`, `pick_device`, `add_device_argument`, `describe_device`, `tensor_bytes`, `format_bytes` |
| `code/scripts/ch03_tensor_tour.py` | Ch 3 | Executed (CPU only) | CLI `--device --size --repeats` |
| `code/examples/ch03/*.py` | Ch 3 | Executed | 9 teaching programs |
| `code/solutions/ch03_pad_and_stack.py` | Ch 3 | Tested, executed | `pad_and_stack(sequences, pad_id=0) -> (ids, mask)` |
| `code/tests/test_devices.py`, `test_ch03_solutions.py` | Ch 3 | 21 tests pass (1 skips only on CUDA machines) | — |
| `tools/audit_chapters.py` | Ch 3 | Executed | Exit 1 on math symbols, filler words, broken links, unexpanded markers |
| `code/llmfp/splits.py` | Ch 4 | Tested | `Splits`, `shuffle_split`, `stable_fraction`, `hash_split`, `deduplicate`, `count_overlap` |
| `code/llmfp/experiment.py` | Ch 4 | Tested | `set_seed`, `file_sha256`, `capture_environment`, `create_run_dir`, `start_run`, `finish_run` |
| `code/llmfp/counting_eval.py` | Ch 4 | Tested | `evaluate_counting_model(model, lines) -> {positions, coverage, accuracy, accuracy_when_covered}` |
| `code/scripts/ch04_make_harbor_corpus.py` | Ch 4 | Executed | Writes `data/tiny/harbor_synth.txt` (seed 0, 3000 lines, SHA-256 a412d79e…) |
| `code/scripts/ch04_evaluate_counting.py` | Ch 4 | Executed | `EvalConfig`; `configs/counting-eval-cpu.toml` |
| `code/solutions/ch04_seed_spread.py`, `ch04_compare_runs.py` | Ch 4 | Executed | — |
| `code/tests/test_splits.py`, `test_experiment.py` | Ch 4 | 21 tests pass | — |
| `code/llmfp/counting_lm.py` | Ch 1 | Tested | `START`, `END`, `CountingModelConfig`, `GenerationResult`, `split_into_words`, `join_words`, `rank_followers`, `CountingLanguageModel` (`train`, `context_for`, `followers_for`, `next_word_candidates`, `generate`, `num_parameters`, `num_contexts`, `vocabulary`, `save`, `load`), `read_lines` |
| `code/scripts/__init__.py` | Ch 1 | — | Makes `scripts` importable as a package |
| `code/scripts/ch01_counting_demo.py` | Ch 1 | Executed | CLI: `--data --context-size --prompt --samples --max-new-words --seed --checkpoint` |
| `code/tests/test_counting_lm.py` | Ch 1 | 18 tests pass | — |
| `code/tests/test_ch01_solutions.py` | Ch 1 | 4 tests pass | — |
| `code/solutions/ch01_backoff.py` | Ch 1 | Tested, executed | `BackoffLanguageModel(config)`; `.last_context_size_used` |
| `code/solutions/ch01_memorization.py` | Ch 1 | Executed | CLI: `--data --samples --max-context-size --seed` |
| `code/README.md` | Ch 1 | — | How to run code |
| `tools/build_book.py` | Ch 1, updated Ch 2 | Executed | Expands `@@FILE@@` and `@@RUN@@` markers in `*.src.md`; RUN uses `code/.venv` when present and captures stderr; `|| true` marks expected failures |
| `code/data/tiny/harbor.txt` | Ch 1 | — | 40 original sentences, written for this book (no third-party license) |

### 5. Dependency versions

| Dependency | Needed from | Version used in testing | Latest on PyPI (checked 2026-10-02) | Pin status |
|---|---|---|---|---|
| Python | Ch 1 | 3.14.4 | — | Book requires 3.12+ (decided; see note) |
| NumPy | Ch 3 | 2.5.3 (venv) | 2.5.3 (requires Python 3.12+) | **Pinned** `==2.5.3` (Ch 2) |
| PyTorch (`torch`) | Ch 3 | 2.14.1+cpu (venv, CPU index) | 2.14.1 | **Pinned** `==2.14.1` (Ch 2). CPU index wheels exist for cp312–cp314; CUDA variants cu126/cu130/cu132 (checked 2026-10-02, untested) |
| pytest | Ch 2 | 9.1.1 | 9.1.1 | **Pinned** `==9.1.1` in `dev` extra (Ch 2) |
| setuptools (build only) | Ch 2 | 78.1.0 | — | `>=77` in `[build-system]` |
| Transitive (torch deps etc.) | Ch 2 | See `requirements/linux-cpu-lock.txt` | — | Locked for Linux CPU |
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
| Ch 2 venv install (Linux CPU) | `python3 -m venv .venv`; torch from CPU index (~48 s); `pip install -e ".[dev]"` | Success; torch stayed `2.14.1+cpu` | 2026-10-02 |
| Ch 2 lock-file install | Fresh venv; `pip install -r requirements/linux-cpu-lock.txt`; `pip install --no-deps -e .` | Identical `pip freeze`; 43 tests passed (at that point) | 2026-10-02 |
| Ch 2 full test suite | `pytest` | 51 passed (18 + 4 Ch 1, 21 config, 8 Ch 2 solutions) | 2026-10-02 |
| Ch 2 bug-catching test | Removed `list(lines)` from backoff; ran `pytest tests/test_ch02_solutions.py`; restored | 1 failed, 7 passed, as intended | 2026-10-02 |
| Ch 2 scripts, examples, error cases | Via `@@RUN@@` markers in chapter build | Output inserted as observed | 2026-10-02 |
| Ch 1 + Ch 2 build | `python3 tools/build_book.py` | All markers expanded; prose scan found no math symbols; all relative links resolve | 2026-10-02 |
| macOS, Windows, GPU install commands | — | **Not executed** (stated in 2.3) | — |
| Ch 3 full test suite | `pytest` | 72 passed | 2026-10-02 |
| Ch 3 examples and tour | Via `@@RUN@@` in build | Output inserted as observed; GPU branch of tour **not executed** | 2026-10-02 |
| Audit of Ch 1–3 | `python3 tools/audit_chapters.py` | ok (Ch 3 "Next" left unlinked until Ch 4 exists) | 2026-10-02 |
| Ch 4 full test suite | `pytest` | 93 passed | 2026-10-02 |
| Ch 4 evaluation runs, reproducibility check, seed spread | Via `@@RUN@@` | Identical reruns showed 0 differences | 2026-10-02 |
| Ch 4 Exercises 2 and 6 claims | Scratch scripts (not in repo) | Numbers quoted in answers as the author's run | 2026-10-02 |
| Audit of Ch 1–4 | `python3 tools/audit_chapters.py` | ok | 2026-10-02 |

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
| Runs overwrite each other in `runs/ch02/` | 2.11 | Ch 4.8: one directory per run, environment capture |
| Config type checks cover only int/float/str/bool/nested dataclasses | 2.9 | Extend if later configs need lists or optional values |
| UTF-8 explained only at the level of byte counts | 2.8 | Ch 8.2–8.3 |
| Matrix multiplication used as a black-box benchmark | 3.12 | Ch 5.3 explains it by behavior |
| Memory = elements times bytes (ignores activations, optimizer state, overhead) | 3.12 | Ch 16.6, 19, 39 |
| Synthetic template corpus (far more regular than real text) | 4.2 | Part 4 real dataset |
| Config validation passes lists through unchecked | 4.7 | Extend `_check_type` if later configs need it |
| Exact-duplicate dedup only | 4.4 | Ch 18.5 near-duplicates |
| Variation judged by spread across salts, no statistics | 4.9 | Ch 37.7 |

### 8. Open questions and decisions

| Decision | Status |
|---|---|
| Default pretraining dataset | Candidate: TinyStories (`roneneldan/TinyStories`, Hub card lists CDLA-Sharing-1.0, checked 2026-10-02). Final choice and license reading in Ch 18 |
| Classification dataset (Ch 26) | Open. The Hub mirror `ucirvine/sms_spam` lists its license as "unknown" (checked 2026-10-02); it will not be used unless the license is confirmed. May write an original harbor dataset instead |
| Pretrained weights for Ch 22 | GPT-2 (`openai-community/gpt2`, Hub lists MIT, checked 2026-10-02) |
| Small instruction models for Parts 5–7 | Candidates (Hub lists Apache-2.0, checked 2026-10-02): `HuggingFaceTB/SmolLM2-135M-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`. Chosen by measurement in Ch 24 |
| Embedding model for Ch 32 | Candidate: `sentence-transformers/all-MiniLM-L6-v2` (Hub lists Apache-2.0, checked 2026-10-02) |
| Serving framework for Ch 40 | Open; decide and verify in Ch 40 |
| Config file format | **Decided and implemented (Ch 2):** TOML for hand-written configs, JSON for machine-written records |
| Environment tool | **Decided (Ch 2):** `venv` + `pip`; other tools mentioned as alternatives |
| `GenerationConfig` location | Exercise 5 suggests keeping it in the script until Ch 17 needs it; revisit in Ch 17 |
