"""Tests for the Chapter 1 counting model.

Run from `code/` with either:
    python -m unittest discover -s tests -v
    python -m pytest tests/test_counting_lm.py      (after Chapter 2 setup)
"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from llmfp.counting_lm import (
    END,
    START,
    CountingLanguageModel,
    CountingModelConfig,
    join_words,
    read_lines,
    split_into_words,
)

TINY_TEXT = [
    "The keeper lit the lamp.",
    "The keeper lit the stove.",
    "The keeper cleaned the lamp.",
]
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "tiny" / "harbor.txt"


def train_tiny(context_size: int = 2) -> CountingLanguageModel:
    model = CountingLanguageModel(CountingModelConfig(context_size=context_size))
    model.train(TINY_TEXT)
    return model


class SplittingTests(unittest.TestCase):
    def test_punctuation_becomes_its_own_word(self) -> None:
        self.assertEqual(split_into_words("Lit at dusk.", lowercase=True), ["lit", "at", "dusk", "."])

    def test_lowercase_is_optional(self) -> None:
        self.assertEqual(split_into_words("The Lamp", lowercase=False), ["The", "Lamp"])

    def test_join_reattaches_punctuation(self) -> None:
        self.assertEqual(join_words(["the", "lamp", "."]), "the lamp.")


class TrainingTests(unittest.TestCase):
    def test_counts_match_hand_calculation(self) -> None:
        model = train_tiny()
        # "keeper lit" is followed by "the" twice; "lit the" by "lamp" once and "stove" once.
        self.assertEqual(model.counts[("keeper", "lit")]["the"], 2)
        self.assertEqual(model.counts[("lit", "the")], {"lamp": 1, "stove": 1})

    def test_sentence_start_and_end_are_recorded(self) -> None:
        model = train_tiny()
        self.assertEqual(model.counts[(START, START)]["the"], 3)
        self.assertEqual(model.counts[("lamp", ".")][END], 2)

    def test_observation_count(self) -> None:
        model = CountingLanguageModel(CountingModelConfig(context_size=2))
        # 6 words + END marker = 7 observations per sentence.
        self.assertEqual(model.train(["The keeper lit the lamp."]), 7)

    def test_empty_lines_are_ignored(self) -> None:
        model = CountingLanguageModel(CountingModelConfig())
        self.assertEqual(model.train(["", "   "]), 0)
        self.assertEqual(model.num_parameters(), 0)

    def test_invalid_context_size_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CountingModelConfig(context_size=0)


class PredictionTests(unittest.TestCase):
    def test_candidates_ranked_by_count_then_alphabetically(self) -> None:
        model = train_tiny()
        self.assertEqual(model.next_word_candidates("the keeper"), [("lit", 2), ("cleaned", 1)])
        self.assertEqual(model.next_word_candidates("lit the"), [("lamp", 1), ("stove", 1)])

    def test_only_last_words_matter(self) -> None:
        model = train_tiny()
        self.assertEqual(
            model.next_word_candidates("anything at all the keeper"),
            model.next_word_candidates("the keeper"),
        )

    def test_unseen_context_gives_no_prediction(self) -> None:
        self.assertEqual(train_tiny().next_word_candidates("purple elephants"), [])

    def test_greedy_generation_is_deterministic(self) -> None:
        result = train_tiny().generate("", greedy=True)
        self.assertEqual(result.text, "the keeper lit the lamp.")
        self.assertEqual(result.stop_reason, "end_marker")

    def test_same_seed_same_samples(self) -> None:
        model = train_tiny()
        rng_a, rng_b = random.Random(7), random.Random(7)
        first = [model.generate("the", rng=rng_a).text for _ in range(5)]
        second = [model.generate("the", rng=rng_b).text for _ in range(5)]
        self.assertEqual(first, second)

    def test_generation_stops_at_unseen_context(self) -> None:
        result = train_tiny().generate("purple elephants")
        self.assertEqual(result.stop_reason, "unseen_context")
        self.assertEqual(result.words, ["purple", "elephants"])

    def test_generation_respects_max_new_words(self) -> None:
        result = train_tiny().generate("", max_new_words=2, greedy=True)
        self.assertEqual(result.words, ["the", "keeper"])
        self.assertEqual(result.stop_reason, "max_new_words")


class CheckpointTests(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        model = train_tiny(context_size=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = CountingLanguageModel.load(path)
        self.assertEqual(loaded.config, model.config)
        self.assertEqual(dict(loaded.counts), dict(model.counts))

    def test_load_rejects_wrong_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.json"
            path.write_text('{"format": "something-else"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                CountingLanguageModel.load(path)


class HarborDatasetTests(unittest.TestCase):
    def test_dataset_loads(self) -> None:
        lines = read_lines(DATA_FILE)
        self.assertEqual(len(lines), 40)
        model = CountingLanguageModel(CountingModelConfig())
        model.train(lines)
        self.assertGreater(model.num_parameters(), 0)


if __name__ == "__main__":
    unittest.main()
