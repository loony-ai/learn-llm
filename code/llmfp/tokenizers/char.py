"""A character-level tokenizer with an unknown token (Chapter 8.7).

The vocabulary is every character seen in training text, sorted, after one
reserved entry: ID 0 is <unk>, used for any character not seen in training.
Encoding unseen characters therefore never fails, but decoding cannot restore
them: they come back as U+FFFD, the replacement character.
"""

from __future__ import annotations

from typing import Any

from llmfp.tokenizers.base import Tokenizer, register

UNKNOWN = "<unk>"
REPLACEMENT = "�"   # what an unknown token decodes to


@register
class CharTokenizer(Tokenizer):
    kind = "char"

    def __init__(self, characters: list[str]) -> None:
        if len(set(characters)) != len(characters):
            raise ValueError("characters must be unique")
        self.characters = list(characters)
        self.index = {character: i + 1 for i, character in enumerate(self.characters)}  # 0 is <unk>

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.characters) + 1

    @property
    def unknown_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [self.index.get(character, self.unknown_id) for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(REPLACEMENT if i == self.unknown_id else self.characters[i - 1] for i in ids)

    def to_dict(self) -> dict[str, Any]:
        return {"characters": self.characters}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharTokenizer":
        return cls(data["characters"])
