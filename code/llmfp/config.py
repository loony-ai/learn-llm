"""Load configuration from TOML files into dataclasses, with command-line overrides (Chapter 2).

A configuration is a frozen dataclass. Its values can come from three places,
applied in this order, later ones winning:

    1. defaults written in the dataclass
    2. a TOML file             (configs/counting-cpu.toml)
    3. command-line overrides  (--set model.context_size=3)

Unknown keys and wrong types are errors, not silent surprises: a misspelled
setting that is quietly ignored is one of the most common ways to run an
experiment different from the one you think you ran.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
import typing
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when configuration data does not match the dataclass it should fill."""


def load_toml(path: str | Path) -> dict[str, Any]:
    """Read a TOML file into nested dictionaries."""
    with open(path, "rb") as file:  # tomllib requires binary mode; TOML is always UTF-8
        return tomllib.load(file)


def parse_value(text: str) -> Any:
    """Interpret an override value the way TOML would, falling back to a plain string.

    "3" -> 3, "0.5" -> 0.5, "true" -> True, "[1, 2]" -> [1, 2], "hello" -> "hello"
    """
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of `data` with "dotted.key=value" overrides applied."""
    result = json.loads(json.dumps(data))  # deep copy; the input is never modified
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key.strip():
            raise ConfigError(f"Override {override!r} must look like key=value or section.key=value")
        *sections, last = key.strip().split(".")
        target = result
        for section in sections:
            target = target.setdefault(section, {})
            if not isinstance(target, dict):
                raise ConfigError(f"Override {override!r}: {section!r} is a value, not a section")
        target[last] = parse_value(raw_value.strip())
    return result


def from_dict(cls: type[T], data: dict[str, Any], where: str = "config") -> T:
    """Build dataclass `cls` from a dictionary, checking names and simple types."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {unknown}. Valid keys: {sorted(known)}")

    values: dict[str, Any] = {}
    for name, value in data.items():
        expected = hints[name]
        path = f"{where}.{name}"
        if dataclasses.is_dataclass(expected):
            if not isinstance(value, dict):
                raise ConfigError(f"{path} must be a section (a TOML table), got {value!r}")
            values[name] = from_dict(expected, value, where=path)
        else:
            values[name] = _check_type(value, expected, path)
    try:
        return cls(**values)
    except TypeError as error:  # e.g. a required field is missing
        raise ConfigError(f"Cannot build {where}: {error}") from error


def _check_type(value: Any, expected: Any, path: str) -> Any:
    """Check the simple types used in configs; anything more complex is passed through."""
    # bool is checked first because Python treats True as an int.
    if expected is bool and not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false, got {value!r}")
    if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ConfigError(f"{path} must be an integer, got {value!r}")
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        return float(value)
    if expected is str and not isinstance(value, str):
        raise ConfigError(f"{path} must be a string, got {value!r}")
    return value


def load_config(cls: type[T], path: str | Path | None = None, overrides: list[str] | None = None) -> T:
    """Defaults, then the TOML file (if given), then overrides -> a validated `cls` instance."""
    data = load_toml(path) if path is not None else {}
    data = apply_overrides(data, overrides or [])
    return from_dict(cls, data)


def to_dict(config: Any) -> dict[str, Any]:
    """Convert a (possibly nested) dataclass config into plain dictionaries."""
    return dataclasses.asdict(config)


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON with stable formatting (sorted keys) so files can be compared and diffed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
