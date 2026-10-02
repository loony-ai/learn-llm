## Hardware Assumptions and Paths

[Back to index](../../README.md)

### The baseline: a CPU-only laptop or desktop

Every chapter can be completed on a CPU-only machine. The book's code is tested on such a machine:

| Item | Test machine (observed 2026-10-02) | Minimum the book plans for |
|---|---|---|
| CPU | 20 logical cores | 4 cores |
| RAM | 14 GiB total | 8 GiB (16 GiB recommended for Parts 5–8) |
| GPU | None | None |
| Disk | n/a | 20 GB free (models and datasets are cached locally) |
| Python | 3.14.4 | 3.12 or newer (NumPy 2.5 requires 3.12+) |
| PyTorch | 2.13.0 (CPU build) | Pinned in Chapter 2 |

On the CPU path, defaults are chosen so that:

- Every test suite runs in under a minute.
- Smoke-test training runs finish in a few minutes.
- The main Part 4 training run finishes in roughly an hour or a few hours. Exact times will be *measured and reported* in Chapter 19, not promised here.
- Pretrained models in Parts 5–8 are small (hundreds of millions of parameters, not billions), so they fit in RAM.

### The optional GPU path

A GPU speeds up training and allows larger experiments. The book supports:

| Option | What it enables | Caveats |
|---|---|---|
| NVIDIA GPU, 8 GB+ VRAM, CUDA | Faster pretraining (Ch 19), larger fine-tunes, QLoRA (Ch 28.5), faster inference engines (Ch 39) | Driver and CUDA versions must match the PyTorch build; covered in Appendix B |
| Apple Silicon (MPS backend) | Faster training and inference than CPU for many operations | Some operations and libraries (notably 4-bit QLoRA tooling) may be unsupported; the book notes each case when it arises |
| Cloud notebook with a GPU | Same as NVIDIA GPU, with no local setup | Sessions time out; save checkpoints to persistent storage (Ch 19.9) |

Every expensive step is preceded by a box stating its expected memory use, rough duration on the test machine, and the CPU alternative.

### How paths are selected in code

From Chapter 3 onward, a single helper chooses the device, and every script accepts `--device`:

```text
--device auto   # default: cuda if available, else mps, else cpu
--device cpu    # force CPU (useful for debugging and reproducibility)
--device cuda   # force NVIDIA GPU
```

Training configurations come in pairs, for example `configs/pretrain-cpu.toml` and `configs/pretrain-gpu.toml`. The CPU version uses a smaller model, shorter context, and fewer steps. Both run the same code.

### What does not fit on modest hardware

Stated here so you are not surprised later:

- Pretraining a model that writes as well as commercial chat models. Out of reach; the book explains why in Chapter 21.10 and Chapter 44.
- Full fine-tuning of models with billions of parameters. Covered conceptually; you will do it on small models and use LoRA for larger ones.
- Preference tuning with PPO at useful scale. Overview only (Chapter 29); DPO is run on a small model.
- Distributed training. Overview only (Chapter 44).
