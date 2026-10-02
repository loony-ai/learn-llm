"""Tests for the Chapter 1 exercise solutions (code/solutions/)."""

from __future__ import annotations

import unittest

from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig
from solutions.ch01_backoff import BackoffLanguageModel

TEXT = ["The keeper lit the lamp.", "The boats left the harbor."]


class BackoffTests(unittest.TestCase):
    def test_backoff_matches_plain_model_on_seen_context(self) -> None:
        config = CountingModelConfig(context_size=3)
        plain, backoff = CountingLanguageModel(config), BackoffLanguageModel(config)
        plain.train(TEXT)
        backoff.train(TEXT)
        self.assertEqual(backoff.next_word_candidates("keeper lit the"), plain.next_word_candidates("keeper lit the"))
        self.assertEqual(backoff.last_context_size_used, 3)

    def test_backoff_uses_shorter_context_when_needed(self) -> None:
        backoff = BackoffLanguageModel(CountingModelConfig(context_size=3))
        backoff.train(TEXT)
        # "boats lit the" never occurred, but "lit the" and "the" did.
        self.assertEqual(backoff.next_word_candidates("boats lit the"), [("lamp", 1)])
        self.assertEqual(backoff.last_context_size_used, 2)

    def test_backoff_gives_up_on_completely_unknown_word(self) -> None:
        backoff = BackoffLanguageModel(CountingModelConfig(context_size=2))
        backoff.train(TEXT)
        self.assertEqual(backoff.next_word_candidates("purple"), [])
        self.assertIsNone(backoff.last_context_size_used)


class IncrementalTrainingTests(unittest.TestCase):
    """Exercise 4: training on A then B gives the same counts as training on A+B."""

    def test_counting_is_order_independent(self) -> None:
        a, b = TEXT[:1], TEXT[1:]
        sequential = CountingLanguageModel(CountingModelConfig())
        sequential.train(a)
        sequential.train(b)
        combined = CountingLanguageModel(CountingModelConfig())
        combined.train(a + b)
        self.assertEqual(dict(sequential.counts), dict(combined.counts))


if __name__ == "__main__":
    unittest.main()
