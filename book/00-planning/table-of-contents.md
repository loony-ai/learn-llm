## Detailed Table of Contents

[Back to index](../../README.md)

Chapter files live under `book/part-N-*/chNN-*.md`. A path in `code/` next to a chapter names the main code it adds. **[Overview only]** marks material you will understand but not implement in full. **[GPU path]** marks steps that need an NVIDIA GPU or a cloud notebook; a CPU alternative is always given.

---

### Part 1: Foundations Beginner Books Often Skip

#### Chapter 1: What a Language Model Is and What It Predicts
File: [ch01-what-a-language-model-predicts.md](../part-1-foundations/ch01-what-a-language-model-predicts.md) · Code: `code/llmfp/counting_lm.py`

- 1.1 The problem: what happens when a chatbot replies?
- 1.2 The family tree: AI, machine learning, deep learning, NLP, language models, LLMs
- 1.3 Rules versus learned behavior
- 1.4 What a model is made of: architecture, parameters, weights, hyperparameters, checkpoints
- 1.5 The model life cycle: training, inference, pretraining, fine-tuning, adaptation
- 1.6 What a language model actually predicts
- 1.7 Generation is a loop
- 1.8 Build it: a counting language model
- 1.9 Code walkthrough
- 1.10 Running the model and reading its output
- 1.11 Experiments: forgetting, repeating, and memorizing
- 1.12 Why next-token prediction becomes useful, and what that does not prove
- 1.13 From a count table to an LLM: the road map
- 1.14 Common mistakes and how to debug them
- 1.15 Recap, concept checks, exercises, answers, checkpoint

#### Chapter 2: Python Foundations and Your Working Environment
File: [ch02-python-foundations-and-environment.md](../part-1-foundations/ch02-python-foundations-and-environment.md) · Code: `code/pyproject.toml`, `code/llmfp/config.py`, `code/scripts/ch02_*.py`, `code/examples/ch02/`

- 2.1 The problem: code that runs today and next month, on your machine and another
- 2.2 Interpreters, virtual environments, packages, and pinned dependencies
- 2.3 Setting up the book repository: `pyproject.toml`, editable installs, pytest
- 2.4 Collections used throughout the book: lists, tuples, dicts, sets, `Counter`, `defaultdict`, dataclasses
- 2.5 Functions, keyword arguments, defaults, and type hints
- 2.6 Classes, methods, properties, and `__call__` (preparing for PyTorch modules)
- 2.7 Iterators and generators: processing data lazily
- 2.8 Files, paths, text encodings, JSON, and TOML
- 2.9 Command-line scripts, configuration, and logging
- 2.10 Writing tests with pytest: fixtures, parametrization, temporary files
- 2.11 Milestone: the Chapter 1 model as an installed, tested package
- 2.12 Common environment errors and how to diagnose them
- 2.13 Recap, concept checks, exercises, answers, checkpoint

#### Chapter 3: Tensors: NumPy and PyTorch as Containers of Numbers
File: [ch03-tensors.md](../part-1-foundations/ch03-tensors.md) · Code: `code/llmfp/devices.py`, `code/scripts/ch03_tensor_tour.py`, `code/examples/ch03/`

- 3.1 The problem: a model processes thousands of numbers at once
- 3.2 From Python lists to arrays: why arrays exist
- 3.3 Shape, dimensions, and axes
- 3.4 Data types: integers, float32, float16, bfloat16, and why they matter
- 3.5 Indexing and slicing
- 3.6 Reshaping, views, transposes, and contiguity
- 3.7 Broadcasting: combining tensors of different shapes
- 3.8 Operations along an axis: sum, mean, max, argmax
- 3.9 Batching: adding a batch dimension
- 3.10 Moving between NumPy and PyTorch
- 3.11 Hardware: CPU, GPU, RAM, VRAM, and devices in PyTorch
- 3.12 Measuring how much memory a tensor uses
- 3.13 Habits for debugging shapes
- 3.14 Recap, concept checks, exercises, answers, checkpoint

#### Chapter 4: Data, Experiments, and Reproducibility
File: [ch04-data-experiments-reproducibility.md](../part-1-foundations/ch04-data-experiments-reproducibility.md) · Code: `code/llmfp/experiment.py`, `code/llmfp/splits.py`, `code/llmfp/counting_eval.py`, `code/scripts/ch04_*.py`

- 4.1 The problem: a result you cannot reproduce is a result you cannot trust
- 4.2 Datasets, examples, inputs, and labels
- 4.3 Training, validation, and test splits, and why there are three
- 4.4 Leakage: how evaluation data sneaks into training
- 4.5 Overfitting and underfitting, observed with the counting model
- 4.6 Random seeds and sources of nondeterminism
- 4.7 Configuration files and command-line overrides
- 4.8 Experiment records: run directories, metadata, environment capture
- 4.9 Milestone: how much of a difference is real?
- 4.10 Common mistakes, recap, concept checks, exercises, answers, checkpoint

#### Chapter 5: Neural Networks Through Behavior and Code
Code: `code/llmfp/nn_basics.py`

- 5.1 The problem: a count table cannot handle contexts it never saw
- 5.2 A single unit: an adjustable scoring rule, in code
- 5.3 Linear layers: many scoring rules at once
- 5.4 Activation functions: what stacking layers needs, observed in code
- 5.5 `nn.Module`, parameters, and the forward pass
- 5.6 Counting and inspecting parameters
- 5.7 Scores (logits) and turning them into a ranked choice (softmax, described by behavior)
- 5.8 A network as a function from tensors to tensors: tracking shapes

#### Chapter 6: How Training Works: Loss, Gradients, and Optimizers
Code: `code/llmfp/training_basics.py`

- 6.1 The problem: how does software adjust millions of numbers sensibly?
- 6.2 Loss: one number that measures how wrong the predictions are (cross-entropy, by behavior)
- 6.3 Nudge-and-observe with a single adjustable number
- 6.4 Gradients: a per-parameter report of which way to move and how sensitive the loss is
- 6.5 Automatic differentiation: PyTorch records operations and computes gradients for you
- 6.6 Optimizers and the learning rate: SGD and AdamW, by behavior
- 6.7 The training loop, line by line
- 6.8 Gradients add up: `zero_grad`, accidental accumulation, and intentional gradient accumulation
- 6.9 Training mode, evaluation mode, and turning off gradient tracking
- 6.10 Experiments: learning rates that are too small, too large, and workable

#### Chapter 7: Project 0: Your First Trained Model, a Next-Character Predictor
Code: `code/llmfp/char_model.py`, `code/scripts/ch07_train_char_model.py`

- 7.1 The problem: replace the count table with something learned
- 7.2 A character vocabulary for the harbor text
- 7.3 Turning text into (context, next character) examples
- 7.4 One-hot inputs: the simplest way to feed characters to a network
- 7.5 The model and its training loop, with validation
- 7.6 Comparing with the counting model on unseen contexts
- 7.7 Sampling text from the trained network
- 7.8 Saving and loading with `state_dict`
- 7.9 First debugging technique: overfit a single batch

---

### Part 2: Turning Text into Model Inputs

#### Chapter 8: Text, Unicode, Bytes, and the Need for Tokens
Code: `code/llmfp/tokenizers/base.py`, `char.py`, `byte.py`

- 8.1 The problem: models consume integers, and text is not integers
- 8.2 Characters, code points, and Unicode
- 8.3 Encodings: how UTF-8 turns characters into bytes
- 8.4 Normalization and invisible differences
- 8.5 Characters, words, and bytes as units: the tradeoffs
- 8.6 Tokens, vocabularies, and token IDs
- 8.7 A common `Tokenizer` interface; character and byte tokenizers
- 8.8 Round-trip tests: decode(encode(text)) must give back the text

#### Chapter 9: Byte-Pair Encoding: Building a Subword Tokenizer (Capstone Project 1)
Code: `code/llmfp/tokenizers/bpe.py`, `code/projects/p1_tokenizer/`

- 9.1 The problem: characters make sequences too long; words make vocabularies too big
- 9.2 The BPE idea, worked by hand
- 9.3 Pre-tokenization: splitting before merging
- 9.4 Training merges
- 9.5 Encoding and decoding with learned merges
- 9.6 Special tokens, unknown tokens, and why byte-level BPE needs no unknown token
- 9.7 Saving and loading a tokenizer
- 9.8 Training your own tokenizer versus using an existing one; comparing against a published tokenizer
- 9.9 How tokenization affects multilingual text, code, numbers, context length, and cost
- 9.10 Capstone Project 1: build and test a tokenizer

#### Chapter 10: Embeddings and Position
Code: `code/llmfp/model/embeddings.py`

- 10.1 The problem: token IDs are labels, not measurements
- 10.2 One-hot inputs revisited, and their cost
- 10.3 Embedding tables: a learned lookup
- 10.4 Vectors as lists of learned features; similarity and cosine similarity, by behavior
- 10.5 Inspecting the embeddings learned in Chapter 7
- 10.6 Why order matters: what a bag of tokens loses
- 10.7 Learned positional embeddings
- 10.8 Preview of other positional methods

#### Chapter 11: Sequences, Batches, and Next-Token Targets
Code: `code/llmfp/data/windows.py`, `code/llmfp/data/collate.py`

- 11.1 The problem: one long text must become many same-shaped examples
- 11.2 Context windows and context length
- 11.3 Inputs and targets: the shift by one, and off-by-one errors
- 11.4 Sliding windows and stride
- 11.5 Batches
- 11.6 Variable lengths: padding, masks, and ignored target positions
- 11.7 Packing several documents into one sequence; document boundaries
- 11.8 PyTorch `Dataset` and `DataLoader`
- 11.9 Milestone: a tested data module

---

### Part 3: Building a Decoder-Only Transformer

#### Chapter 12: A Tour of Transformer Architectures
Code: `code/llmfp/model/config.py`

- 12.1 The problem: fixed windows cannot use long context flexibly
- 12.2 Sequence models before transformers, briefly
- 12.3 The transformer idea at a glance
- 12.4 Encoder-only, decoder-only, and encoder-decoder models: what each is for
- 12.5 Why this book builds a decoder-only model
- 12.6 Map of the GPT-style model, with the chapter that builds each box
- 12.7 The `GPTConfig` object

#### Chapter 13: Attention From Scratch
Code: `code/llmfp/model/attention.py`

- 13.1 The problem: which earlier tokens should influence this position?
- 13.2 Attention as a weighted lookup: the analogy and its limits
- 13.3 Self-attention with no trainable weights
- 13.4 Queries, keys, and values: three learned projections
- 13.5 Attention scores and attention weights
- 13.6 Causal attention: no looking at future tokens
- 13.7 Padding masks and combining masks
- 13.8 A batched single-head attention module
- 13.9 Tests: shapes, masks, and a causality test

#### Chapter 14: Multi-Head Attention
Code: `code/llmfp/model/attention.py` (extended)

- 14.1 The problem: one attention pattern per position is limiting
- 14.2 Heads: several smaller attention operations side by side
- 14.3 Splitting and merging heads with reshape and transpose
- 14.4 The output projection
- 14.5 Readable versus optimized: comparing with PyTorch's built-in attention function
- 14.6 Shape trace and tests

#### Chapter 15: Completing the Transformer Block
Code: `code/llmfp/model/layers.py`, `positions.py`, `block.py`

- 15.1 The problem: attention mixes positions but does not transform each one
- 15.2 Feed-forward layers and activation functions (ReLU, GELU, gated variants)
- 15.3 Residual connections: keep the input, add a change
- 15.4 Normalization: LayerNorm and RMSNorm, and what they keep stable
- 15.5 Pre-norm versus post-norm
- 15.6 Dropout
- 15.7 Rotary positional embeddings (RoPE), through code and observed behavior
- 15.8 The `TransformerBlock` class

#### Chapter 16: Assembling a GPT-Style Model (Capstone Project 2)
Code: `code/llmfp/model/gpt.py`, `code/projects/p2_transformer/`

- 16.1 The problem: turning parts into a working model
- 16.2 Embeddings, block stack, final normalization, output head
- 16.3 Logits and choosing a token
- 16.4 Parameter initialization: why starting values matter
- 16.5 Weight tying
- 16.6 Counting parameters and estimating memory
- 16.7 Tracing one batch through every component
- 16.8 Tests: shapes, causality, masks, gradient flow, determinism
- 16.9 Common implementation bugs and how to locate them
- 16.10 Capstone Project 2: implement a small decoder-only transformer

#### Chapter 17: Generating Text and KV Caching
Code: `code/llmfp/generation/generate.py`, `code/llmfp/model/kv_cache.py`

- 17.1 The problem: from next-token predictor to text generator
- 17.2 The generation loop, greedy first
- 17.3 Cropping context when text exceeds the window
- 17.4 Why naive generation repeats work
- 17.5 KV caching: keeping keys and values from earlier steps
- 17.6 Implementing the cache and testing that outputs are unchanged
- 17.7 Measuring the speedup on your hardware
- 17.8 Cache memory cost and pitfalls

---

### Part 4: Training and Generating Text

#### Chapter 18: Pretraining Data: Licensing, Quality, and Pipelines
Code: `code/llmfp/data/pipeline.py`, `code/scripts/ch18_prepare_data.py`

- 18.1 The problem: the data determines what the model can learn
- 18.2 Choosing a licensed dataset; reading dataset cards; provenance
- 18.3 The book's default dataset and why it was chosen
- 18.4 Cleaning: encodings, boilerplate, filtering
- 18.5 Deduplication: exact duplicates and near-duplicates
- 18.6 Contamination between splits and evaluation sets
- 18.7 A reproducible pipeline: download, record hashes, clean, split, tokenize, shard
- 18.8 Storing token IDs efficiently on disk

#### Chapter 19: The Pretraining Loop
Code: `code/llmfp/training/trainer.py`, `schedules.py`, `checkpoint.py`

- 19.1 The problem: train for hours safely and know whether it is working
- 19.2 Structure of a pretraining loop
- 19.3 Interpreting loss values without equations
- 19.4 Training and validation curves
- 19.5 AdamW and weight decay, by behavior
- 19.6 Learning-rate schedules and warmup
- 19.7 Gradient clipping
- 19.8 Batch size, tokens per step, and gradient accumulation
- 19.9 Checkpoints: model, optimizer, schedule, random state, tokenizer, configuration; resuming exactly
- 19.10 Mixed precision and device handling
- 19.11 Running the CPU and GPU configurations

#### Chapter 20: Debugging Training
Code: `code/scripts/ch20_bug_lab.py`

- 20.1 The problem: the run finishes but the model has not learned
- 20.2 An order of operations for debugging
- 20.3 Overfitting a tiny batch, revisited
- 20.4 Loss that does not decrease
- 20.5 Loss spikes and invalid values (NaN and infinity)
- 20.6 Validation that looks too good: leakage and bugs
- 20.7 Slow training: is it data loading or compute?
- 20.8 Bug lab: find the seeded bugs

#### Chapter 21: Decoding Strategies and Honest Expectations (Capstone Project 3)
Code: `code/llmfp/generation/decoding.py`, `code/projects/p3_train_and_analyze/`

- 21.1 The problem: the same model can produce very different text
- 21.2 Greedy decoding
- 21.3 Sampling
- 21.4 Temperature
- 21.5 Top-k and top-p
- 21.6 Stop conditions and length limits
- 21.7 Repetition and repetition penalties
- 21.8 Comparing settings systematically
- 21.9 Fluent but wrong: why it happens
- 21.10 What a tiny model on modest hardware can and cannot do
- 21.11 Capstone Project 3: train, resume from a checkpoint, analyze limitations

---

### Part 5: Working with Pretrained Models

#### Chapter 22: Loading Pretrained Weights into Your Own Model
Code: `code/llmfp/pretrained/gpt2_loader.py`

- 22.1 The problem: you cannot pretrain at scale, but published weights exist
- 22.2 Reading configuration files and model cards
- 22.3 Weight files, the safetensors format, state dicts, and parameter names
- 22.4 Mapping GPT-2 weights onto our architecture
- 22.5 Verifying the result against a reference implementation
- 22.6 Licenses for model weights

#### Chapter 23: Using Established Model Libraries
Code: `code/llmfp/pretrained/hf_models.py`

- 23.1 The problem: real work spans many architectures
- 23.2 The Hugging Face Hub and `transformers`, mapped to what you built
- 23.3 Downloading, caching, pinning revisions, offline use, saving locally
- 23.4 Base, instruction, and chat models
- 23.5 Chat templates and special tokens
- 23.6 Generation settings in libraries
- 23.7 Model licenses, dataset licenses, and usage restrictions

#### Chapter 24: Choosing and Running Models on Your Hardware
Code: `code/llmfp/pretrained/select.py`, `code/scripts/ch24_compare_models.py`

- 24.1 The problem: which model should you use?
- 24.2 Estimating memory from parameter count and data type
- 24.3 A small task benchmark for comparing candidates
- 24.4 Quantization: what it does, and its tradeoffs
- 24.5 Running quantized models on CPU and GPU
- 24.6 Measuring the quality cost of quantization
- 24.7 Writing a model selection record

---

### Part 6: Adapting Models

#### Chapter 25: Choosing an Adaptation Strategy
- 25.1 The problem: the model is not good enough at your task
- 25.2 Prompting, retrieval, fine-tuning, continued pretraining: what each changes
- 25.3 A decision guide with costs and risks
- 25.4 Baselines before adaptation
- 25.5 Defining the narrow task used in this part

#### Chapter 26: Classification Fine-Tuning
Code: `code/llmfp/adapt/classification.py`

- 26.1 The problem: you need a label, not prose
- 26.2 Replacing the output head
- 26.3 Choosing and checking a labeled dataset
- 26.4 Which layers to train; freezing
- 26.5 Training and evaluating: accuracy, confusion matrices, per-class results
- 26.6 Comparing against a baseline

#### Chapter 27: Supervised Instruction Tuning
Code: `code/llmfp/adapt/instruction.py`

- 27.1 The problem: a base model continues text instead of following instructions
- 27.2 Instruction dataset schemas
- 27.3 Instruction data quality
- 27.4 Chat formatting
- 27.5 Loss masking: learning from responses only
- 27.6 Packing and truncation
- 27.7 Full fine-tuning of a small model
- 27.8 Catastrophic forgetting and overfitting checks

#### Chapter 28: Parameter-Efficient Fine-Tuning: LoRA and QLoRA (Capstone Project 4)
Code: `code/llmfp/adapt/lora.py`, `code/projects/p4_adapt/`

- 28.1 The problem: full fine-tuning needs memory for every parameter and its optimizer state
- 28.2 LoRA: small trainable add-ons beside frozen weights
- 28.3 Implementing LoRA on our GPT
- 28.4 Using the `peft` library
- 28.5 QLoRA: LoRA on a quantized base model [GPU path]
- 28.6 Choosing rank, target layers, and learning rate
- 28.7 Merging, saving, and loading adapters
- 28.8 Evaluating whether adaptation helped
- 28.9 Capstone Project 4: adapt a pretrained model to a narrow task

#### Chapter 29: Preference Tuning: RLHF, Reward Models, PPO, and DPO
Code: `code/llmfp/adapt/dpo.py`

- 29.1 The problem: several answers are acceptable, but some are preferred
- 29.2 Preference data
- 29.3 The RLHF pipeline [Overview only]
- 29.4 Reward models [Overview only]
- 29.5 PPO in plain language [Overview only]
- 29.6 DPO: learning directly from preferred and rejected pairs
- 29.7 A small runnable DPO experiment
- 29.8 Limits: reward hacking and biased preference data

---

### Part 7: Building Useful LLM Applications

#### Chapter 30: Prompts and Context Management
Code: `code/llmfp/apps/llm_client.py`, `prompts.py`

- 30.1 The problem: the same model gives better or worse results depending on its input
- 30.2 Messages, roles, and the instruction hierarchy
- 30.3 Clear instructions, delimiters, and few-shot examples
- 30.4 Context budgets: counting tokens, truncation, and summarization
- 30.5 Prompt templates as versioned code
- 30.6 The `LLMClient` interface: local and hosted backends

#### Chapter 31: Structured Outputs and Failure Handling
Code: `code/llmfp/apps/structured.py`

- 31.1 The problem: programs need data, not prose
- 31.2 Schemas and validation
- 31.3 Parsing model output safely
- 31.4 Retries with error feedback, and their limits
- 31.5 Constrained decoding [Overview with a small demo]
- 31.6 Failure handling and fallbacks

#### Chapter 32: Embeddings and Search
Code: `code/llmfp/apps/embeddings.py`, `chunking.py`, `vector_store.py`, `rerank.py`

- 32.1 The problem: finding the passages that answer a question
- 32.2 Embedding models versus generative models
- 32.3 Keyword search, by behavior
- 32.4 Chunking strategies and metadata
- 32.5 A vector store from scratch, then a library
- 32.6 Hybrid search and reranking

#### Chapter 33: Retrieval-Augmented Generation with Citations (Capstone Project 5)
Code: `code/llmfp/apps/rag.py`, `code/projects/p5_rag/`

- 33.1 The problem: answer from specific documents and show where the answer came from
- 33.2 The Harbor Handbook corpus
- 33.3 The RAG pipeline end to end
- 33.4 Citations and refusing to answer when the sources do not support an answer
- 33.5 Evaluating retrieval separately from generation
- 33.6 Evaluating answers: correctness, grounding, citation accuracy
- 33.7 Failure analysis: what RAG does and does not fix
- 33.8 Capstone Project 5

#### Chapter 34: Tool Calling, Workflows, and Agents
Code: `code/llmfp/apps/tools.py`

- 34.1 The problem: some answers require actions or live data
- 34.2 Tool schemas and how a model proposes a call
- 34.3 Validating and executing calls; returning results
- 34.4 Workflows versus agents
- 34.5 When an agent makes things worse

#### Chapter 35: A Constrained Tool-Using Assistant (Capstone Project 6)
Code: `code/llmfp/apps/agent.py`, `memory.py`, `code/projects/p6_tool_assistant/`

- 35.1 The problem: useful actions with bounded risk
- 35.2 Permissions and allowlists
- 35.3 Approval gates for consequential actions
- 35.4 Sandboxing and limits on steps, time, and cost
- 35.5 Audit logs
- 35.6 Memory implementations and their privacy and reliability tradeoffs
- 35.7 Capstone Project 6

#### Chapter 36: Security for LLM Applications
Code: `code/llmfp/apps/security.py`

- 36.1 The problem: text from anywhere can steer the model
- 36.2 Direct and indirect prompt injection
- 36.3 Treating retrieved content and tool output as untrusted
- 36.4 Exfiltration through tools and links
- 36.5 Handling secrets
- 36.6 Defense in depth and security tests

---

### Part 8: Evaluation, Reliability, and Production Engineering

#### Chapter 37: Designing Evaluations
Code: `code/llmfp/eval/`

- 37.1 The problem: "it seemed good when I tried it" is not evidence
- 37.2 Task-specific evaluation datasets
- 37.3 Baselines and comparisons
- 37.4 Training-data contamination and benchmark limitations
- 37.5 Human evaluation
- 37.6 Model-based judges and their limitations
- 37.7 Variation between runs, and how many examples are enough, by behavior

#### Chapter 38: Correctness, Grounding, Safety, and Regression Testing
Code: `code/llmfp/eval/regression.py`

- 38.1 The problem: a change that fixes one case breaks three others
- 38.2 Quality dimensions: correctness, grounding, instruction following, safety
- 38.3 Hallucination analysis
- 38.4 Uncertainty handling and abstention
- 38.5 Regression suites and running them in CI

#### Chapter 39: Inference Performance
Code: `code/scripts/ch39_benchmark_inference.py`

- 39.1 The problem: correct but too slow or too expensive
- 39.2 Latency, time to first token, throughput, memory
- 39.3 Measuring properly
- 39.4 KV caching revisited
- 39.5 Batching: static and continuous
- 39.6 Streaming
- 39.7 Optimized attention, intuitively
- 39.8 Inference engines [Overview with a local demo]

#### Chapter 40: Serving LLM Applications
Code: `code/llmfp/serving/`

- 40.1 The problem: many users, one model
- 40.2 An HTTP API with streaming responses
- 40.3 Concurrency
- 40.4 Cancellation and timeouts
- 40.5 Backpressure and queue limits
- 40.6 Load testing

#### Chapter 41: Operating in Production (Capstone Project 7)
Code: `code/llmfp/serving/monitoring.py`, `code/projects/p7_serve/`

- 41.1 The problem: it worked in testing and fails at 2 a.m.
- 41.2 Monitoring and tracing
- 41.3 Privacy-conscious logging
- 41.4 Incident debugging
- 41.5 Versioning models, prompts, data, and evaluation suites
- 41.6 Deployment and rollback
- 41.7 Cost estimation from measured workloads
- 41.8 Capstone Project 7

#### Chapter 42: Responsible Deployment
- 42.1 The problem: harms that tests for correctness do not catch
- 42.2 Bias: where it comes from and how to look for it
- 42.3 Privacy
- 42.4 Copyright and licensing questions (not legal advice)
- 42.5 Documentation: model cards and system cards
- 42.6 A pre-launch review

---

### Part 9: Advanced Orientation and Professional Practice

#### Chapter 43: Architectures Beyond the Basics
- 43.1 Mixture-of-experts models [Overview with a toy router]
- 43.2 Distillation [Overview with a small runnable example]
- 43.3 Multimodal models [Overview only]
- 43.4 Long-context techniques and their limitations [Overview only]

#### Chapter 44: Reasoning Models and Distributed Training
- 44.1 Reasoning-oriented models and inference-time compute [Overview with a best-of-n demo]
- 44.2 What we know and do not know about these models
- 44.3 Distributed training: the problems it solves [Overview only]
- 44.4 Data, tensor, and pipeline parallelism; sharded optimizers [Overview only]

#### Chapter 45: Professional Practice
- 45.1 How to read a research paper without the math
- 45.2 How to inspect an unfamiliar model repository
- 45.3 Reproducing a small experiment
- 45.4 Assessing claims about new models and techniques
- 45.5 Finding your gaps and choosing the next experiment

---

### Appendices

#### Appendix A: Glossary
#### Appendix B: Environment Setup and Troubleshooting
#### Appendix C: Tensor-Shape Reference
#### Appendix D: Common Training and Inference Errors
#### Appendix E: Evaluation Checklist
#### Appendix F: Practical Skills Rubric
#### Appendix G: Further Reading
