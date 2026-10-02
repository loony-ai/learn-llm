"""Tests for llmfp.nn_basics (Chapter 5)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from llmfp.nn_basics import TinyMLP, count_parameters, format_parameter_table, parameter_table, shape_trace


def test_output_shape_for_2d_and_3d_inputs():
    model = TinyMLP(3, 5, 7)
    assert model(torch.randn(2, 3)).shape == (2, 7)
    assert model(torch.randn(2, 4, 3)).shape == (2, 4, 7)


def test_parameter_count_matches_hand_count():
    # hidden: 5 rows of 3 weights + 5 biases; output: 7 rows of 5 weights + 7 biases
    model = TinyMLP(3, 5, 7)
    assert count_parameters(model) == (5 * 3 + 5) + (7 * 5 + 7)


def test_trainable_only_excludes_frozen_parameters():
    model = TinyMLP(3, 5, 7)
    model.hidden.weight.requires_grad_(False)
    assert count_parameters(model, trainable_only=True) == count_parameters(model) - 15


def test_parameter_table_names_and_shapes():
    rows = parameter_table(TinyMLP(3, 5, 7))
    assert [(r.name, r.shape) for r in rows] == [
        ("hidden.weight", (5, 3)),
        ("hidden.bias", (5,)),
        ("output.weight", (7, 5)),
        ("output.bias", (7,)),
    ]
    assert "total" in format_parameter_table(TinyMLP(3, 5, 7))


def test_same_seed_same_initialization():
    torch.manual_seed(0)
    first = TinyMLP(3, 5, 7)
    torch.manual_seed(0)
    second = TinyMLP(3, 5, 7)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def test_unknown_activation_rejected():
    with pytest.raises(ValueError):
        TinyMLP(3, 5, 7, activation="swish-ish")


def test_no_activation_network_is_equivalent_to_one_linear_layer():
    torch.manual_seed(0)
    model = TinyMLP(4, 6, 2, activation="none")
    merged = nn.Linear(4, 2)
    with torch.no_grad():
        merged.weight.copy_(model.output.weight @ model.hidden.weight)
        merged.bias.copy_(model.output.weight @ model.hidden.bias + model.output.bias)
    x = torch.randn(10, 4)
    assert torch.allclose(model(x), merged(x), atol=1e-5)


def test_shape_trace_records_every_layer_in_order():
    trace = shape_trace(TinyMLP(3, 5, 7), torch.randn(2, 3))
    assert trace == [("hidden", (2, 5)), ("activation", (2, 5)), ("output", (2, 7)), ("(output)", (2, 7))]


def test_shape_trace_removes_hooks_even_on_error():
    model = TinyMLP(3, 5, 7)
    with pytest.raises(RuntimeError):
        shape_trace(model, torch.randn(2, 4))  # wrong number of input features
    assert all(len(m._forward_hooks) == 0 for m in model.modules())


def test_softmax_rows_are_positive_and_total_one():
    shares = torch.softmax(TinyMLP(3, 5, 7)(torch.randn(4, 3)), dim=-1)
    assert bool((shares > 0).all())
    assert torch.allclose(shares.sum(dim=-1), torch.ones(4))
