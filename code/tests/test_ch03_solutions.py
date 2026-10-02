"""Tests for the Chapter 3 exercise solutions."""

from __future__ import annotations

import pytest
import torch

from solutions.ch03_pad_and_stack import pad_and_stack


def test_shapes_and_values():
    ids, mask = pad_and_stack([[5, 9, 2], [7]], pad_id=0)
    assert ids.shape == mask.shape == (2, 3)
    assert ids.tolist() == [[5, 9, 2], [7, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert ids.dtype == torch.int64 and mask.dtype == torch.bool


def test_pad_id_that_is_also_a_real_token_is_distinguished_by_the_mask():
    ids, mask = pad_and_stack([[0, 0], [0]], pad_id=0)
    assert ids.tolist() == [[0, 0], [0, 0]]
    assert mask.tolist() == [[True, True], [True, False]]


def test_equal_lengths_need_no_padding():
    ids, mask = pad_and_stack([[1, 2], [3, 4]])
    assert torch.equal(ids, torch.tensor([[1, 2], [3, 4]]))
    assert bool(mask.all())


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        pad_and_stack([])
