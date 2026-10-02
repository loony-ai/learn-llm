"""The interface every tokenizer in the book follows, and saving/loading (Chapter 8.7)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

FORMAT = "llmfp-tokenizer-v1"
_REGISTRY: dict[str, type["Tokenizer"]] = {}


def register(cls: type["Tokenizer"]) -> type["Tokenizer"]:
    """Class decorator: lets load_tokenizer rebuild this kind of tokenizer from its saved name."""
    _REGISTRY[cls.kind] = cls
    return cls


class Tokenizer(ABC):
    """Text in, integer IDs out, and back again.

    Every subclass sets `kind` (a short name stored in saved files) and implements
    encode, decode, vocab_size, to_dict, and from_dict.
    """

    kind: str = "abstract"

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """How many distinct IDs this tokenizer can produce: valid IDs are 0 .. vocab_size - 1."""

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this tokenizer, as JSON-compatible data."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tokenizer": ...

    def round_trips(self, text: str) -> bool:
        """True if decoding the encoding gives back exactly the original text."""
        return self.decode(self.encode(text)) == text


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "kind": tokenizer.kind, **tokenizer.to_dict()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a saved tokenizer (format={data.get('format')!r})")
    kind = data.pop("kind")
    data.pop("format")
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer kind {kind!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[kind].from_dict(data)
