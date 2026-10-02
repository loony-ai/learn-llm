"""Chapter 2.10: a deliberately failing test, to show how pytest reports failures.

This file is in examples/, not tests/, so a plain `pytest` run never collects it.
Run it explicitly from `code/`:
    pytest examples/ch02/test_failure_demo.py
"""


def make_inputs_and_targets(token_ids: list[int]) -> tuple[list[int], list[int]]:
    """Inputs are all tokens but the last; each target is the token that follows its input."""
    return token_ids[:-1], token_ids[:-1]  # BUG: targets should be token_ids[1:]


def test_targets_are_inputs_shifted_by_one():
    inputs, targets = make_inputs_and_targets([10, 11, 12, 13])
    assert inputs == [10, 11, 12]
    assert targets == [11, 12, 13]
