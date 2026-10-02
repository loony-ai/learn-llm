# Companion code

Code for *Large Language Models From First Principles*. Run every command from this `code/` directory, with the `.venv` environment active (Chapter 2).

## Setup (Chapter 2)

```bash
python3 -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\Activate.ps1
# Linux without an NVIDIA GPU (tested path):
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# macOS / Windows: python -m pip install torch==2.14.1   (GPU builds: see Chapter 2.3)
python -m pip install -e ".[dev]"
python -m scripts.ch02_check_env              # verify versions
pytest                                        # run all tests
```

Linux CPU alternative: `python -m pip install -r requirements/linux-cpu-lock.txt && python -m pip install --no-deps -e .`

## Entry points by chapter

| Chapter | Command |
|---|---|
| 1 | `python -m scripts.ch01_counting_demo` |
| 2 | `python -m scripts.ch02_train_counting --set model.context_size=3` |
| 3 | `python -m scripts.ch03_tensor_tour --device auto` |
| 4 | `python -m scripts.ch04_make_harbor_corpus` and `python -m scripts.ch04_evaluate_counting` |
| 5 | `python -m scripts.ch05_untrained_network` |
| 6 | `python -m scripts.ch06_train_band` and `python -m scripts.ch06_learning_rates` |
| 7 | `python -m scripts.ch07_train_char_model` (add `--set overfit_one_batch=true` for the debugging mode) |
| 8 | `python -m scripts.ch08_compare_units` |
| 9 | `python -m scripts.ch09_build_corpus`, `python -m scripts.ch09_train_bpe`, `python -m scripts.ch09_compare_tokenizers` (Project 1: `projects/p1_tokenizer/`) |
| 10 | `python -m scripts.ch10_train_embedding_model` |
| 11 | `python -m scripts.ch11_build_batches` |

Teaching examples live in `examples/chNN/` (run with `python examples/chNN/<name>.py`); exercise solutions in `solutions/` (run with `python -m solutions.<name>`).

Generated files go in `runs/`, one directory per run, which can be deleted at any time.
