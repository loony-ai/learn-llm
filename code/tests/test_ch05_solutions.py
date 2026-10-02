"""Tests for the Chapter 5 exercise solutions."""

from __future__ import annotations

import torch

from solutions.ch05_two_bands import build_two_band_network


def outputs(model, values):
    with torch.no_grad():
        return model(torch.tensor(values).unsqueeze(-1)).squeeze(-1).tolist()


def test_peaks_at_band_centers():
    assert outputs(build_two_band_network(), [1.0, 3.0]) == [1.0, 1.0]


def test_zero_outside_bands():
    assert outputs(build_two_band_network(), [-1.0, 0.0, 2.0, 4.0, 5.0]) == [0.0] * 5


def test_half_height_halfway_up_each_side():
    for value in outputs(build_two_band_network(), [0.75, 1.25, 2.75, 3.25]):
        assert abs(value - 0.5) < 1e-6
