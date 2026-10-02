"""Tests for llmfp.training_basics and llmfp.toy_data (Chapter 6), and Chapter 6 solutions."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from llmfp.nn_basics import TinyMLP
from llmfp.toy_data import make_band_data
from llmfp.training_basics import evaluate, fit, iterate_minibatches, train_step
from solutions.ch06_accumulated_step import accumulated_train_step


def test_band_data_shapes_labels_and_reproducibility():
    inputs, labels = make_band_data(100, seed=3)
    assert inputs.shape == (100, 1) and labels.shape == (100,)
    assert inputs.dtype == torch.float32 and labels.dtype == torch.int64
    inside = (inputs[:, 0] >= 1.5) & (inputs[:, 0] <= 2.5)
    assert torch.equal(labels.bool(), inside)
    assert torch.equal(make_band_data(100, seed=3)[0], inputs)


def test_minibatches_cover_every_example_once():
    inputs, targets = torch.arange(10).unsqueeze(-1), torch.arange(10)
    batches = list(iterate_minibatches(inputs, targets, 4, generator=torch.Generator().manual_seed(0)))
    assert [len(t) for _, t in batches] == [4, 4, 2]
    assert sorted(torch.cat([t for _, t in batches]).tolist()) == list(range(10))
    assert all(torch.equal(x[:, 0], t) for x, t in batches)  # inputs stay paired with targets


def test_minibatches_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        list(iterate_minibatches(torch.zeros(3, 1), torch.zeros(4), 2))


def test_train_step_changes_parameters_and_clears_old_gradients():
    torch.manual_seed(0)
    model = TinyMLP(1, 4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.full_like(p, 1000.0)  # stale gradient that must be cleared
    inputs, targets = make_band_data(16)
    train_step(model, inputs, targets, nn.CrossEntropyLoss(), optimizer)
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad.abs().max() < 1000 for p in model.parameters())


def test_evaluate_uses_eval_mode_restores_mode_and_tracks_no_gradients():
    model = nn.Sequential(nn.Linear(1, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    inputs, targets = make_band_data(50)
    model.train()
    first = evaluate(model, inputs, targets, nn.CrossEntropyLoss())
    second = evaluate(model, inputs, targets, nn.CrossEntropyLoss())
    assert first == second            # dropout was off, so results are repeatable
    assert model.training             # caller's mode restored
    assert all(p.grad is None for p in model.parameters())


def test_fit_learns_the_band():
    torch.manual_seed(0)
    model = TinyMLP(1, 16, 2)
    history = fit(
        model, make_band_data(2000, seed=0), make_band_data(500, seed=1), nn.CrossEntropyLoss(),
        torch.optim.AdamW(model.parameters(), lr=0.01), epochs=15, batch_size=32,
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["val_accuracy"] > 0.9


def test_fit_stops_when_loss_is_not_finite():
    torch.manual_seed(0)
    model = TinyMLP(1, 16, 2)
    with torch.no_grad():
        model.output.bias.fill_(float("nan"))
    history = fit(
        model, make_band_data(64), make_band_data(64, seed=1), nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1), epochs=5, batch_size=32,
    )
    assert len(history) == 1


@pytest.mark.parametrize("micro_batches", [2, 3, 5])
def test_accumulated_step_matches_full_batch_step(micro_batches):
    inputs, targets = make_band_data(31, seed=2)  # 31 does not divide evenly into 2, 3, or 5 chunks
    models = []
    for chunks in (1, micro_batches):
        torch.manual_seed(0)
        model = TinyMLP(1, 8, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        accumulated_train_step(model, inputs, targets, nn.CrossEntropyLoss(), optimizer, chunks)
        models.append(model)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(models[0].parameters(), models[1].parameters()))
