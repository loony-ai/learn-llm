# Companion code

Code for *Large Language Models From First Principles*. Run every command from this `code/` directory.

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

## Running chapter code

```bash
python -m scripts.ch01_counting_demo                          # Chapter 1
python -m scripts.ch02_train_counting --set model.context_size=3   # Chapter 2
python examples/ch02/generators_tour.py                       # Chapter 2 examples
python -m solutions.ch01_backoff                              # exercise solutions
```

Generated files go in `runs/`, which can be deleted at any time.
