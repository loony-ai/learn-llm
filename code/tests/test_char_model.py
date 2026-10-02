"""Tests for llmfp.char_model (Chapter 7) and the Chapter 7 solutions."""

from __future__ import annotations

import pytest
import torch

from llmfp.char_model import (
    CharMLP,
    CharModelConfig,
    CharVocabulary,
    counting_baseline,
    load_checkpoint,
    make_examples,
    sample_text,
    save_checkpoint,
)
from solutions.ch07_fact_check import classify
from solutions.ch07_off_by_one import make_examples_with_bug


def test_vocabulary_is_sorted_and_round_trips():
    vocabulary = CharVocabulary.build("the keeper\n")
    assert vocabulary.characters == ["\n", " ", "e", "h", "k", "p", "r", "t"]
    assert vocabulary.decode(vocabulary.encode("keep the\n")) == "keep the\n"
    assert CharVocabulary.from_dict(vocabulary.to_dict()).characters == vocabulary.characters


def test_unknown_character_gives_clear_error():
    with pytest.raises(ValueError, match="'z'"):
        CharVocabulary.build("abc").encode("abz")


def test_make_examples_pairs_each_window_with_the_next_id():
    contexts, targets = make_examples([10, 11, 12, 13, 14], context_size=3)
    assert contexts.tolist() == [[10, 11, 12], [11, 12, 13]]
    assert targets.tolist() == [13, 14]
    assert contexts.dtype == targets.dtype == torch.int64


def test_make_examples_rejects_too_short_input():
    with pytest.raises(ValueError):
        make_examples([1, 2, 3], context_size=3)


def test_buggy_targets_are_the_last_character_of_each_window():
    contexts, targets = make_examples_with_bug([10, 11, 12, 13, 14], context_size=3)
    assert torch.equal(targets, contexts[:, -1])   # the bug: target already in the input


def test_model_shapes_and_one_hot_input():
    model = CharMLP(CharModelConfig(vocab_size=7, context_size=4, hidden=16))
    assert model(torch.randint(0, 7, (5, 4))).shape == (5, 7)
    assert model.hidden.in_features == 4 * 7


def test_counting_baseline_seen_and_correct():
    train_x = torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4]])
    train_y = torch.tensor([5, 5, 6, 7])
    result = counting_baseline(train_x, train_y, torch.tensor([[1, 2], [3, 4], [9, 9]]), torch.tensor([5, 0, 1]))
    assert result["seen"].tolist() == [True, True, False]
    assert result["correct"].tolist() == [True, False, False]
    assert result["coverage"] == pytest.approx(2 / 3)


def test_checkpoint_round_trip_gives_identical_outputs(tmp_path):
    torch.manual_seed(0)
    vocabulary = CharVocabulary.build("the keeper lit the lamp.\n")
    model = CharMLP(CharModelConfig(vocabulary.size, 4, 16))
    save_checkpoint(tmp_path / "ckpt", model, vocabulary)
    reloaded, reloaded_vocabulary = load_checkpoint(tmp_path / "ckpt")
    x = torch.randint(0, vocabulary.size, (10, 4))
    assert torch.equal(model(x), reloaded(x))
    assert reloaded_vocabulary.characters == vocabulary.characters


def test_sampling_is_reproducible_and_pads_short_prompts():
    torch.manual_seed(0)
    vocabulary = CharVocabulary.build("the keeper lit the lamp.\n")
    model = CharMLP(CharModelConfig(vocabulary.size, 6, 16))
    first = sample_text(model, vocabulary, "the", 20, torch.Generator().manual_seed(1))
    second = sample_text(model, vocabulary, "the", 20, torch.Generator().manual_seed(1))
    assert first == second and first.startswith("the") and len(first) == 23


@pytest.mark.parametrize(
    ("sentence", "label"),
    [
        ("The keeper lit the lamp at dusk.", "well-formed and true"),
        ("At night the gulls followed the boats.", "well-formed and true"),
        ("The keeper warned the fishers in the rain.", "well-formed but false"),
        ("The gulls lit the lamp and fed the gulls at noon.", "well-formed but false"),
        ("The keeper lit the lamp.", "not well-formed"),
        ("The keeper rangaten bell at dawn.", "not well-formed"),
    ],
)
def test_fact_check_classifier(sentence, label):
    assert classify(sentence) == label
