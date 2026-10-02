"""Tests for llmfp.splits (Chapter 4)."""

from __future__ import annotations

import pytest

from llmfp.splits import count_overlap, deduplicate, hash_split, shuffle_split, stable_fraction

ITEMS = [f"sentence {i}" for i in range(1000)]


def test_shuffle_split_sizes_and_coverage():
    splits = shuffle_split(ITEMS, (0.8, 0.1, 0.1), seed=0)
    assert splits.sizes() == {"train": 800, "validation": 100, "test": 100}
    assert sorted(splits.train + splits.validation + splits.test) == sorted(ITEMS)


def test_shuffle_split_is_reproducible_and_seed_dependent():
    assert shuffle_split(ITEMS, seed=1) == shuffle_split(ITEMS, seed=1)
    assert shuffle_split(ITEMS, seed=1).validation != shuffle_split(ITEMS, seed=2).validation


def test_shuffle_split_does_not_modify_input():
    items = list(ITEMS)
    shuffle_split(items, seed=3)
    assert items == ITEMS


@pytest.mark.parametrize("fractions", [(0.5, 0.5), (0.8, 0.3, 0.1), (1.2, -0.1, -0.1)])
def test_bad_fractions_rejected(fractions):
    with pytest.raises(ValueError):
        shuffle_split(ITEMS, fractions)


def test_stable_fraction_is_fixed_and_in_range():
    assert stable_fraction("the keeper") == stable_fraction("the keeper")
    assert 0.0 <= stable_fraction("the keeper") < 1.0
    assert stable_fraction("the keeper", salt="a") != stable_fraction("the keeper", salt="b")


def test_hash_split_puts_duplicates_together():
    items = ["a", "b", "a", "c", "a", "b"] * 50
    splits = hash_split(items)
    assert count_overlap(splits.train, splits.validation) == 0
    assert count_overlap(splits.train, splits.test) == 0


def test_hash_split_is_stable_when_data_grows():
    small = hash_split(ITEMS[:500])
    large = hash_split(ITEMS)
    assert set(small.validation) <= set(large.validation)


def test_hash_split_sizes_are_approximately_right():
    sizes = hash_split(ITEMS).sizes()
    assert 750 <= sizes["train"] <= 850


def test_hash_split_with_custom_key_groups_items():
    items = [("keeper", 1), ("keeper", 2), ("gulls", 3), ("gulls", 4)] * 25
    splits = hash_split(items, key=lambda item: item[0])
    groups = [{name for name, _ in part} for part in (splits.train, splits.validation, splits.test)]
    assert sum(len(g) for g in groups) == 2  # each name in exactly one split


def test_deduplicate_keeps_first_occurrence_in_order():
    assert deduplicate(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert deduplicate(["A", "a"], key=str.lower) == ["A"]


def test_count_overlap():
    assert count_overlap(["a", "b"], ["b", "b", "c"]) == 2
