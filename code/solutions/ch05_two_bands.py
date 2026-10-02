"""Chapter 5, Exercise 6 (suggested solution): a hand-built network that detects two bands.

Output close to 1 for inputs near 1 and near 3, close to 0 elsewhere. Each band
uses three hidden ReLU units, exactly like the single band in why_activations.py:
units that switch on at the left edge, the center, and the right edge of the band,
combined with output weights 1, -2, 1.

Run from `code/`:  python -m solutions.ch05_two_bands
"""

from __future__ import annotations

import torch

from llmfp.nn_basics import TinyMLP


def build_two_band_network(centers: tuple[float, float] = (1.0, 3.0), width: float = 0.5) -> TinyMLP:
    """Each hidden unit stays at zero until the input passes its edge, then rises steadily.

    The output adds the left-edge and right-edge units and subtracts the center unit twice,
    which makes a peak at the center that falls back to zero at both edges.
    """
    model = TinyMLP(1, 6, 1, activation="relu")
    edges = [edge for center in centers for edge in (center - width, center, center + width)]
    with torch.no_grad():
        model.hidden.weight.copy_(torch.full((6, 1), 1.0 / width))
        model.hidden.bias.copy_(torch.tensor([-edge / width for edge in edges]))
        model.output.weight.copy_(torch.tensor([[1.0, -2.0, 1.0, 1.0, -2.0, 1.0]]))
        model.output.bias.zero_()
    return model


def main() -> None:
    model = build_two_band_network()
    probe = torch.arange(0.0, 4.51, 0.25).unsqueeze(-1)
    with torch.no_grad():
        outputs = model(probe).squeeze(-1)
    for x, y in zip(probe.squeeze(-1).tolist(), outputs.tolist()):
        print(f"input {x:4.2f} -> {y:4.2f} {'#' * round(y * 20)}")


if __name__ == "__main__":
    main()
