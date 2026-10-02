# Companion code

Code for *Large Language Models From First Principles*. Run every command from this `code/` directory.

## Chapter 1 (standard-library Python only)

```bash
python3 -m unittest discover -s tests -v       # run all tests
python3 -m scripts.ch01_counting_demo          # train, inspect, save, reload, generate
python3 -m scripts.ch01_counting_demo --context-size 1 --prompt "the"
python3 -m solutions.ch01_backoff              # Exercise 5 solution
python3 -m solutions.ch01_memorization         # Exercise 6 solution
```

Scripts are run with `-m` so that this directory is on Python's import path and `import llmfp` works without installation. Chapter 2 adds `pyproject.toml` and an editable install.

Generated files go in `runs/`, which can be deleted at any time.
