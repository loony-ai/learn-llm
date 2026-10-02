"""Tests for llmfp.experiment and llmfp.counting_eval (Chapter 4)."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from llmfp.config import load_json
from llmfp.counting_eval import evaluate_counting_model
from llmfp.counting_lm import CountingLanguageModel, CountingModelConfig
from llmfp.experiment import capture_environment, create_run_dir, file_sha256, finish_run, set_seed, start_run


def draws():
    return random.random(), float(np.random.rand()), torch.rand(1).item()


def test_set_seed_makes_all_generators_repeat():
    set_seed(7)
    first = draws()
    set_seed(7)
    assert draws() == first


def test_file_sha256_changes_with_content(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("one\n", encoding="utf-8")
    before = file_sha256(path)
    assert before == file_sha256(path)
    path.write_text("one\ntwo\n", encoding="utf-8")
    assert file_sha256(path) != before


def test_capture_environment_has_key_fields():
    environment = capture_environment()
    for key in ("python", "platform", "packages", "git_commit", "command", "started_utc"):
        assert key in environment
    assert environment["packages"]["torch"] is not None


def test_run_dirs_are_unique(tmp_path):
    first = create_run_dir(tmp_path, "exp")
    second = create_run_dir(tmp_path, "exp")
    assert first != second and first.is_dir() and second.is_dir()


@dataclass(frozen=True)
class TinyConfig:
    seed: int = 3


def test_start_and_finish_run_write_records(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello\n", encoding="utf-8")
    run_dir = start_run(tmp_path / "runs", "exp", TinyConfig(), data_files=[data])
    logging.getLogger("test").warning("inside the run")
    finish_run(run_dir, {"accuracy": 0.5})
    logging.getLogger("test").warning("after the run")

    assert load_json(run_dir / "config.json") == {"seed": 3}
    assert load_json(run_dir / "environment.json")["data_files"] == {str(data): file_sha256(data)}
    assert load_json(run_dir / "metrics.json") == {"accuracy": 0.5}
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "inside the run" in log and "after the run" not in log


def test_counting_eval_on_training_text():
    model = CountingLanguageModel(CountingModelConfig(context_size=2))
    model.train(["The keeper lit the lamp."])
    metrics = evaluate_counting_model(model, ["The keeper lit the lamp."])
    assert metrics == {"positions": 7, "coverage": 1.0, "accuracy": 1.0, "accuracy_when_covered": 1.0}


def test_counting_eval_counts_unseen_contexts_as_wrong():
    model = CountingLanguageModel(CountingModelConfig(context_size=2))
    model.train(["The keeper lit the lamp."])
    metrics = evaluate_counting_model(model, ["The gulls lit the lamp."])
    # Positions: (s,s)->the ok, (s,the)->gulls wrong, (the,gulls) unseen, (gulls,lit) unseen,
    # (lit,the)->lamp ok, (the,lamp)->. ok, (lamp,.)->end ok
    assert metrics["positions"] == 7
    assert metrics["coverage"] == pytest.approx(5 / 7)
    assert metrics["accuracy"] == pytest.approx(4 / 7)
    assert metrics["accuracy_when_covered"] == pytest.approx(4 / 5)


def test_counting_eval_rejects_empty_input():
    model = CountingLanguageModel(CountingModelConfig())
    with pytest.raises(ValueError):
        evaluate_counting_model(model, ["", "  "])
