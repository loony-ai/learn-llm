"""Small neural-network building blocks and inspection tools (Chapter 5).

    TinyMLP            a configurable two-layer network: Linear -> activation -> Linear
    count_parameters   how many adjustable numbers a module holds
    parameter_table    name, shape, and count of every parameter tensor
    shape_trace        run a forward pass and record the output shape of every submodule
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,  # passes values through unchanged
}


class TinyMLP(nn.Module):
    """A multi-layer perceptron with one hidden layer.

    input (batch, in_features)
      -> Linear  -> (batch, hidden_features)
      -> activation
      -> Linear  -> (batch, out_features)   one score per output
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, activation: str = "relu") -> None:
        super().__init__()  # required: sets up PyTorch's parameter tracking
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        self.hidden = nn.Linear(in_features, hidden_features)
        self.activation = ACTIVATIONS[activation]()
        self.output = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(x)))


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Total number of individual numbers in a module's parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    shape: tuple[int, ...]
    count: int


def parameter_table(module: nn.Module) -> list[ParameterInfo]:
    return [ParameterInfo(name, tuple(p.shape), p.numel()) for name, p in module.named_parameters()]


def format_parameter_table(module: nn.Module) -> str:
    rows = parameter_table(module)
    width = max([len("parameter")] + [len(row.name) for row in rows])
    lines = [f"{'parameter':<{width}}  {'shape':<14} {'count':>10}"]
    lines += [f"{row.name:<{width}}  {str(row.shape):<14} {row.count:>10,}" for row in rows]
    lines.append(f"{'total':<{width}}  {'':<14} {count_parameters(module):>10,}")
    return "\n".join(lines)


def shape_trace(module: nn.Module, *inputs: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Run `module` on `inputs` and return (submodule name, output shape) in execution order.

    Uses forward hooks: functions PyTorch calls after a submodule's forward runs.
    The hooks are removed afterwards, even if the forward pass raises an error.
    """
    trace: list[tuple[str, tuple[int, ...]]] = []
    handles = []
    for name, submodule in module.named_modules():
        if name == "":
            continue  # the top-level module itself is recorded last, below

        def hook(_module, _inputs, output, name=name):
            if isinstance(output, torch.Tensor):
                trace.append((name, tuple(output.shape)))

        handles.append(submodule.register_forward_hook(hook))
    try:
        result = module(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    if isinstance(result, torch.Tensor):
        trace.append(("(output)", tuple(result.shape)))
    return trace
