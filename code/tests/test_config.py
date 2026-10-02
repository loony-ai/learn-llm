"""Tests for llmfp.config (Chapter 2). Written in pytest style."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llmfp.config import (
    ConfigError,
    apply_overrides,
    from_dict,
    load_config,
    load_json,
    parse_value,
    save_json,
    to_dict,
)
from llmfp.counting_lm import CountingModelConfig
from scripts.ch02_train_counting import CountingRunConfig

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "counting-cpu.toml"


@dataclass(frozen=True)
class Inner:
    size: int = 2
    rate: float = 0.5


@dataclass(frozen=True)
class Outer:
    name: str = "run"
    verbose: bool = False
    inner: Inner = field(default_factory=Inner)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3", 3),
        ("0.5", 0.5),
        ("true", True),
        ('"quoted"', "quoted"),
        ("[1, 2]", [1, 2]),
        ("plain words", "plain words"),
    ],
)
def test_parse_value(text, expected):
    assert parse_value(text) == expected


def test_defaults_when_no_data():
    assert from_dict(Outer, {}) == Outer()


def test_nested_section_keeps_unspecified_defaults():
    config = from_dict(Outer, {"inner": {"size": 4}})
    assert config.inner == Inner(size=4, rate=0.5)


def test_int_accepted_where_float_expected():
    config = from_dict(Outer, {"inner": {"rate": 1}})
    assert config.inner.rate == 1.0
    assert isinstance(config.inner.rate, float)


@pytest.mark.parametrize(
    "data",
    [
        {"nmae": "typo"},                 # unknown key
        {"inner": {"szie": 3}},           # unknown nested key
        {"verbose": "yes"},               # wrong type: string for bool
        {"inner": {"size": "3"}},         # wrong type: string for int
        {"inner": {"size": True}},        # bool is not accepted as an int
        {"inner": 5},                     # value where a section is expected
    ],
)
def test_bad_data_is_rejected(data):
    with pytest.raises(ConfigError):
        from_dict(Outer, data)


def test_overrides_do_not_modify_input():
    original = {"inner": {"size": 2}}
    updated = apply_overrides(original, ["inner.size=8", "name=test"])
    assert original == {"inner": {"size": 2}}
    assert updated == {"inner": {"size": 8}, "name": "test"}


def test_malformed_override_is_rejected():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals_sign"])


def test_load_config_layers_file_then_overrides(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('name = "from-file"\n[inner]\nsize = 3\n', encoding="utf-8")
    config = load_config(Outer, path, ["inner.size=9"])
    assert config == Outer(name="from-file", inner=Inner(size=9))


def test_json_round_trip(tmp_path):
    config = Outer(name="x", inner=Inner(size=7))
    path = tmp_path / "sub" / "config.json"
    save_json(to_dict(config), path)
    assert from_dict(Outer, load_json(path)) == config


def test_project_config_file_matches_dataclass_defaults():
    # The shipped TOML file and the dataclass defaults should describe the same run.
    assert load_config(CountingRunConfig, CONFIG_FILE) == CountingRunConfig()


def test_counting_model_config_validation_still_applies():
    with pytest.raises(ValueError):
        from_dict(CountingRunConfig, {"model": {"context_size": 0}})
    assert from_dict(CountingRunConfig, {"model": {"context_size": 3}}).model == CountingModelConfig(context_size=3)
