"""A byte-level tokenizer: every UTF-8 byte value is a token (Chapter 8.7).

There are exactly 256 possible byte values, so the vocabulary is fixed, and any
text in any language encodes without an "unknown" token. The price: most
non-English characters take 2-4 tokens, so sequences get long.
"""

from __future__ import annotations

from typing import Any

from llmfp.tokenizers.base import Tokenizer, register


@register
class ByteTokenizer(Tokenizer):
    kind = "byte"

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        # errors="replace": an incomplete or invalid byte sequence becomes U+FFFD (the
        # replacement character) instead of crashing. This matters when a model generates
        # the first byte of a multi-byte character but not the rest.
        return bytes(ids).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ByteTokenizer":
        return cls()
