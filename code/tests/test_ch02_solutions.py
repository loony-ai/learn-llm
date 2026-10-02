"""Tests for the Chapter 2 exercise solutions (code/solutions/)."""

from __future__ import annotations

import pytest

from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig, split_into_words
from solutions.ch01_backoff import BackoffLanguageModel
from solutions.ch02_iter_sentences import iter_sentences


def test_iter_sentences_reads_files_in_order_and_skips_blanks(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("one\n\n  two  \n", encoding="utf-8")
    second.write_text("three\n", encoding="utf-8")
    assert list(iter_sentences([first, second])) == ["one", "two", "three"]


def test_iter_sentences_is_lazy(tmp_path):
    existing = tmp_path / "a.txt"
    existing.write_text("one\n", encoding="utf-8")
    sentences = iter_sentences([existing, tmp_path / "missing.txt"])
    assert next(sentences) == "one"          # the missing file has not been opened yet
    with pytest.raises(FileNotFoundError):   # ...until the consumer reaches it
        next(sentences)


def test_training_from_generator_equals_training_from_list(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("The keeper lit the lamp.\nThe boats left.\n", encoding="utf-8")
    from_generator = CountingLanguageModel(CountingModelConfig())
    from_generator.train(iter_sentences([path]))
    from_list = CountingLanguageModel(CountingModelConfig())
    from_list.train(path.read_text(encoding="utf-8").splitlines())
    assert dict(from_generator.counts) == dict(from_list.counts)


def test_backoff_model_trains_every_level_from_a_one_shot_generator(tmp_path):
    """Exercise 4b: this test fails if BackoffLanguageModel.train forgets list(lines)."""
    path = tmp_path / "data.txt"
    path.write_text("The keeper lit the lamp.\n", encoding="utf-8")
    model = BackoffLanguageModel(CountingModelConfig(context_size=3))
    model.train(iter_sentences([path]))
    assert model.num_contexts() > 0
    assert all(shorter.num_contexts() > 0 for shorter in model.shorter_models)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The keeper's lamp", ["the", "keeper", "'", "s", "lamp"]),
        ("Pier No. 2, 3.5 m", ["pier", "no", ".", "2", ",", "3", ".", "5", "m"]),
        ("café naïve", ["café", "naïve"]),
        ("well-known", ["well", "-", "known"]),
    ],
)
def test_split_into_words_edge_cases(text, expected):
    """Exercise 3: pins down current behavior, including the surprising cases."""
    assert split_into_words(text, lowercase=True) == expected
