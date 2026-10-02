# Large Language Models From First Principles: A Math-Free Practical Guide

A hands-on book for software developers who want to build, train, adapt, evaluate, debug, and deploy language models, without equations.

You will write a small language model yourself in Python and PyTorch, train it, and then use what you learned to work with real pretrained models and build applications around them. Every concept is explained before it is used. Every milestone has runnable code, tests, and exercises.

> **An honest promise.** Reading this book will not make you an expert. Doing the projects can make you *practically competent*: able to implement, read, train, adapt, evaluate, and debug real systems. Theoretical mastery of the field requires mathematics this book deliberately leaves out. Where that limit matters, the text says so.

### How to use this book

1. Read [the learning journey](book/00-planning/learning-journey.md) (5 minutes).
2. Start [Chapter 1](book/part-1-foundations/ch01-what-a-language-model-predicts.md). It needs only Python, with no installation beyond the language itself.
3. Run every script and test in [`code/`](code/). Do the exercises. Code you only read builds far less skill than code you run, break, and fix.
4. Keep an experiment notebook (a plain text file is fine). Chapter 4 explains what to record.

### Contents

The full table of contents, with every numbered section, is in [table-of-contents.md](book/00-planning/table-of-contents.md).

| Part | Chapters | Status |
|---|---|---|
| **Part 1: Foundations beginner books often skip** | [1 What a language model is and what it predicts](book/part-1-foundations/ch01-what-a-language-model-predicts.md) · [2 Python foundations and your environment](book/part-1-foundations/ch02-python-foundations-and-environment.md) · [3 Tensors](book/part-1-foundations/ch03-tensors.md) · [4 Data, experiments, reproducibility](book/part-1-foundations/ch04-data-experiments-reproducibility.md) · [5 Neural networks through behavior and code](book/part-1-foundations/ch05-neural-networks.md) · [6 How training works](book/part-1-foundations/ch06-how-training-works.md) · [7 Project 0: your first trained model](book/part-1-foundations/ch07-project-0-char-model.md) | **Complete** (Ch 1–7) |
| **Part 2: Turning text into model inputs** | [8 Text, Unicode, bytes, tokens](book/part-2-text-to-inputs/ch08-text-unicode-bytes-tokens.md) · [9 Byte-pair encoding (Project 1)](book/part-2-text-to-inputs/ch09-byte-pair-encoding.md) · 10 Embeddings and position · 11 Sequences, batches, targets | Ch 8–9 written |
| **Part 3: Building a decoder-only transformer** | 12 Architecture tour · 13 Attention from scratch · 14 Multi-head attention · 15 Completing the block · 16 Assembling a GPT-style model (Project 2) · 17 Generation and KV caching | Planned |
| **Part 4: Training and generating text** | 18 Pretraining data · 19 The pretraining loop · 20 Debugging training · 21 Decoding and honest expectations (Project 3) | Planned |
| **Part 5: Working with pretrained models** | 22 Loading pretrained weights into your model · 23 Established model libraries · 24 Choosing and running models on your hardware | Planned |
| **Part 6: Adapting models** | 25 Choosing an adaptation strategy · 26 Classification fine-tuning · 27 Instruction tuning · 28 LoRA and QLoRA (Project 4) · 29 Preference tuning | Planned |
| **Part 7: Building useful LLM applications** | 30 Prompts and context · 31 Structured outputs · 32 Embeddings and search · 33 RAG with citations (Project 5) · 34 Tool calling, workflows, agents · 35 A constrained tool-using assistant (Project 6) · 36 Security | Planned |
| **Part 8: Evaluation, reliability, production** | 37 Designing evaluations · 38 Correctness, grounding, regression testing · 39 Inference performance · 40 Serving · 41 Operating in production (Project 7) · 42 Responsible deployment | Planned |
| **Part 9: Advanced orientation and practice** | 43 Architectures beyond the basics · 44 Reasoning models and distributed training · 45 Professional practice | Planned |
| **Appendices** | A Glossary · B Environment setup · C Tensor shapes · D Common errors · E Evaluation checklist · F Skills rubric · G Further reading | Planned |

### Planning documents

- [Learning journey](book/00-planning/learning-journey.md)
- [Detailed table of contents](book/00-planning/table-of-contents.md)
- [Prerequisite map](book/00-planning/prerequisite-map.md)
- [Repository structure](book/00-planning/repository-structure.md)
- [Capstone map](book/00-planning/capstone-map.md)
- [Hardware assumptions and paths](book/00-planning/hardware-paths.md)
- [Coverage audit](book/00-planning/coverage-audit.md)
- [Editorial ledger](book/00-planning/editorial-ledger.md)

### Conventions

- **Zero mathematics.** No equations or notation anywhere. Code may do arithmetic because software needs it; the prose explains what the code *does*, not a formula behind it.
- **Labels on output.** `Observed output` means the author ran it and pasted the result. `Illustrative output` means a representative example that was not run.
- **Labels on scope.** Chapters and sections marked **Overview only** explain ideas you will not implement in full here.
- **Analogies are labeled** and followed by what the software actually does.
