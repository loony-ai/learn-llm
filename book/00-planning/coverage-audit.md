## Coverage Audit

[Back to index](../../README.md)

This audit lists the gaps that beginner LLM material commonly leaves, and where this book closes each one. It also maps every required topic from the book specification to a chapter, so nothing is dropped as the book grows.

### Common missing foundations, and where they are closed

| Common gap | Typical symptom for the learner | Where this book closes it |
|---|---|---|
| "Model", "parameter", "weights", and "checkpoint" are used before they are defined | Cannot tell code, architecture, and saved numbers apart | 1.4, with a model whose parameters you can read in a JSON file |
| Next-token prediction is stated but never shown | "Predicts the next word" sounds either trivial or magical | 1.6–1.11: a working counting model you can inspect |
| Capabilities of LLMs are overstated, or dismissed | Confusion between fluency and correctness | 1.12, revisited in 21.9 and 38.3 |
| Python packaging and environments are assumed | Import errors, version conflicts, "works on my machine" | Chapter 2, Appendix B |
| Tensors appear with no explanation of shape and axes | Shape errors are copied around until they stop crashing | Chapter 3, Appendix C, shape traces in 14.6 and 16.7 |
| Validation and test splits are mentioned without explaining leakage | Inflated results that do not hold up | 4.3–4.4, 18.6, 20.6, 37.4 |
| Training is explained only through calculus, or not at all | The training loop is cargo-culted | Chapter 6, entirely by observed behavior and code |
| `zero_grad`, `eval()`, `no_grad` are used as rituals | Silent bugs: accumulating gradients, dropout during evaluation | 6.8–6.9 with experiments showing what goes wrong |
| The first neural network is a transformer | Too many new ideas at once | Chapter 7 trains a small network before any transformer |
| Tokenization is a black box | Surprise at costs, multilingual behavior, code handling | Chapters 8–9, with a BPE you write yourself |
| Embeddings are called "meaning vectors" with no mechanism | Treating token IDs as numbers with magnitude | 10.1–10.4 |
| The input/target shift is shown once and never tested | Off-by-one errors that still produce a decreasing loss | 11.3, with a test that catches the error |
| Attention is introduced with formulas or loose metaphors only | Cannot implement or debug masks | Chapters 13–14: analogy labeled, then code, then tests |
| Residuals, normalization, and dropout are listed, not motivated | Cannot reason about why training is unstable | 15.3–15.6, each with an experiment |
| No tests for model code | Bugs found only by bad samples | 13.9, 14.6, 16.8: causality, mask, gradient-flow tests |
| Checkpoints save only weights | Cannot resume training correctly | 19.9: full state, with a resume test |
| Debugging training is left to intuition | Abandoned runs, misread curves | Chapter 20 and its bug lab |
| Decoding knobs are copied without explanation | Repetition and nonsense output | Chapter 21 with side-by-side comparisons |
| Libraries are introduced before the internals | Library code feels like magic | Chapter 22 loads GPT-2 weights into your own model first |
| Chat templates and special tokens are skipped | Instruction models behave strangely | 23.4–23.5, 27.4 |
| Fine-tuning is presented as the default fix | Expensive work that prompting or retrieval would have solved | Chapter 25 |
| Loss masking during instruction tuning is unexplained | The model learns to imitate prompts | 27.5 |
| RAG is presented as a cure for hallucination | Unsupported answers with confident citations | 33.4–33.7, 38.3 |
| Agents are presented as always better | Fragile, expensive, unsafe systems | 34.4–34.5, Chapter 35 |
| Security is left out | Prompt injection and leaked secrets | Chapter 36 |
| Evaluation means "try a few prompts" | No regression protection | Chapters 37–38 |
| Production concerns are ignored | Timeouts, overload, unbounded cost | Chapters 39–41 |

### Specification coverage

Each required topic from the book specification and the section that covers it. "✓" means delivered; blank means planned.

#### Part 1

| Required topic | Section(s) | Delivered |
|---|---|---|
| AI, ML, DL, NLP, LMs, LLMs and how they relate | 1.2–1.3 | ✓ |
| Models, parameters, weights, architectures, checkpoints, hyperparameters | 1.4 | ✓ |
| Training, inference, pretraining, fine-tuning, adaptation | 1.5 | ✓ |
| What a language model predicts | 1.6–1.7 | ✓ |
| Why next-token prediction can produce useful capabilities; what it does not prove | 1.12 | ✓ |
| Python: collections, functions, classes, iterators, generators, files, environments, packages | 2.2–2.9 | |
| NumPy and PyTorch prerequisites | Chapter 3 | |
| Tensors: shapes, dimensions, axes, dtypes, devices, indexing, slicing, reshaping, broadcasting, batching | 3.3–3.11 | |
| CPU, GPU, RAM, VRAM, hardware constraints | 3.11–3.12 | |
| Reproducibility, seeds, configuration, experiment records | 4.6–4.8 | |
| Datasets, examples, labels, splits, leakage, overfitting | 4.2–4.5 | |
| Neural networks through behavior and code | Chapter 5 | |
| Forward pass, loss, backward pass, autodiff, gradients, optimizers, learning rates, training loops | Chapter 6 | |
| Gradient accumulation vs accidental accumulation | 6.8 | |
| Evaluation mode, training mode, disabling gradient tracking | 6.9 | |
| Runnable environment and tiny first training project before transformers | Chapters 2, 7 | Partial: Ch 1 runs with standard-library Python |

#### Parts 2–9

| Part | Required topics | Covered in |
|---|---|---|
| 2 | Encoding, Unicode, bytes, characters, words, tokens; why tokenization; char/word/subword/byte tokenization; BPE implementation; vocabulary, IDs, special and unknown tokens; training vs existing tokenizer; round trips; multilingual, code, context length, cost | Chapters 8–9 |
| 2 | Embeddings, tables, IDs as identifiers; positional information | Chapter 10 |
| 2 | Sequences, context windows, sliding windows, batches, padding, packing, masks; inputs and targets; off-by-one errors | Chapter 11 |
| 3 | Architecture tour; why decoder-only | Chapter 12 |
| 3 | Attention, self-attention, causal attention; Q/K/V; scores and weights; causal and padding masks | Chapter 13 |
| 3 | Multi-head attention; optimized library operations | Chapter 14 |
| 3 | Feed-forward, activations, residuals, normalization, dropout; positional embeddings and RoPE | Chapter 15 |
| 3 | Blocks, output heads, logits, token selection; initialization; weight tying; full model; batch trace; tests; common bugs | Chapter 16 |
| 3 | KV caching after basic generation | Chapter 17 |
| 4 | Licensed dataset; provenance, quality, cleaning, dedup, contamination; reproducible pipeline | Chapter 18 |
| 4 | Pretraining loop; loss interpretation; curves; optimizers, schedules, warmup, clipping, batch sizes, accumulation; checkpoints and resume; tokenizer/config state; mixed precision; devices | Chapter 19 |
| 4 | Debugging instability, invalid values, poor learning, suspicious validation; overfitting a tiny batch | Chapter 20 |
| 4 | Greedy, sampling, temperature, top-k, top-p, stopping, repetition; comparisons; fluent but wrong; realistic expectations | Chapter 21 |
| 5 | Configs and model cards; loading tokenizers and weights; matching names | Chapter 22 |
| 5 | Established libraries; base/instruction/chat; chat templates; licenses; downloading, caching, saving, running locally | Chapter 23 |
| 5 | Selecting models by measurement and hardware; quantization | Chapter 24 |
| 6 | When to prompt, retrieve, fine-tune, or continue pretraining | Chapter 25 |
| 6 | Classification fine-tuning | Chapter 26 |
| 6 | Full fine-tuning; instruction tuning; schemas; data quality; loss masking; chat formatting; packing; truncation; forgetting; overfitting | Chapter 27 |
| 6 | PEFT, LoRA, QLoRA; evaluating adaptation | Chapter 28 |
| 6 | RLHF, reward models, PPO, DPO overview; runnable vs conceptual separation | Chapter 29 |
| 7 | Prompt construction, instruction hierarchy, few-shot, context management | Chapter 30 |
| 7 | Structured outputs, validation, retries, failure handling | Chapter 31 |
| 7 | Embedding models; search, chunking, metadata, vector stores, retrieval, reranking | Chapter 32 |
| 7 | Complete RAG with citations; retrieval evaluated separately | Chapter 33 |
| 7 | Tool calling, schemas, validation; workflows vs agents | Chapter 34 |
| 7 | Constrained tool-using app; permissions, approvals, sandboxing, limits; memory tradeoffs | Chapter 35 |
| 7 | Prompt injection, untrusted content, secrets | Chapter 36 |
| 8 | Task-specific eval sets, baselines, contamination, benchmark limits, human eval, model judges | Chapter 37 |
| 8 | Correctness, grounding, instruction following, safety, regression tests, hallucination, uncertainty | Chapter 38 |
| 8 | Latency, throughput, TTFT, memory; KV caching, batching, streaming, engines; optimized attention | Chapter 39 |
| 8 | Serving APIs, concurrency, cancellation, timeouts, backpressure | Chapter 40 |
| 8 | Monitoring, tracing, logging, incidents; versioning; deployment, rollback, reproducibility; cost from measurements | Chapter 41 |
| 8 | Bias, privacy, copyright, responsible deployment | Chapter 42 |
| 9 | MoE, distillation, multimodal, long context | Chapter 43 |
| 9 | Reasoning models, inference-time compute, distributed training | Chapter 44 |
| 9 | Reading papers, inspecting repos, reproducing experiments, assessing claims, identifying gaps | Chapter 45 |
| Appendices | Glossary, setup, tensor shapes, errors, eval checklist, skills rubric, further reading | Appendices A–G |
