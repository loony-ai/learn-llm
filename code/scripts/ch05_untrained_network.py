"""Chapter 5 milestone: an untrained network that scores every word in the harbor vocabulary.

The network receives a list of numbers describing a context and returns one score
per vocabulary word. The inputs here are random placeholders: Chapter 7 shows how
to turn real text into inputs, and Chapter 6 shows how training makes the scores
mean something. Until then, the scores are arbitrary, but they exist for EVERY
input, which is exactly what the counting model could not offer.

Run from `code/`:
    python -m scripts.ch05_untrained_network
    python -m scripts.ch05_untrained_network --seed 1 --hidden 64
"""

from __future__ import annotations

import argparse

import torch

from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig, read_lines
from llmfp.devices import add_device_argument, format_bytes, pick_device, tensor_bytes
from llmfp.experiment import set_seed
from llmfp.nn_basics import TinyMLP, format_parameter_table, shape_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/tiny/harbor.txt")
    parser.add_argument("--context-features", type=int, default=16, help="numbers describing one context")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    add_device_argument(parser)
    args = parser.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)

    # The vocabulary: every word the Chapter 1 model can output, sorted for stable IDs.
    counting = CountingLanguageModel(CountingModelConfig())
    counting.train(read_lines(args.data))
    vocabulary = sorted(counting.vocabulary())

    model = TinyMLP(args.context_features, args.hidden, len(vocabulary)).to(device)
    print(f"Vocabulary: {len(vocabulary)} words. Model:\n{model}\n")
    print(format_parameter_table(model))
    parameter_bytes = sum(tensor_bytes(p) for p in model.parameters())
    print(f"Parameter memory: {format_bytes(parameter_bytes)}\n")

    contexts = torch.randn(args.batch, args.context_features, device=device)  # placeholder inputs
    print("Shape trace for a batch of", args.batch, "contexts:")
    for name, shape in shape_trace(model, contexts):
        print(f"  {name:<12} {shape}")

    with torch.no_grad():  # no training here; Chapter 6 explains this line
        logits = model(contexts)
        shares = torch.softmax(logits, dim=-1)
    top = shares[0].topk(5)
    print("\nTop 5 words for context 0 (untrained, so arbitrary):")
    for share, index in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {vocabulary[index]:<10} {share:.3f}")
    print(f"Even share would be {1 / len(vocabulary):.3f} per word; largest share here {shares.max():.3f}.")


if __name__ == "__main__":
    main()
