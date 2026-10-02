"""Tests for llmfp.tokenizers: base, char, and byte tokenizers (Chapter 8)."""

from __future__ import annotations

import random

import pytest

from llmfp.tokenizers import ByteTokenizer, CharTokenizer, load_tokenizer, save_tokenizer
from llmfp.tokenizers.char import REPLACEMENT

TRICKY_TEXTS = [
    "",
    "The keeper lit the lamp.",
    "Café 🐟 at the pier",
    "é vs é",
    "\U0001F468‍\U0001F469‍\U0001F467 family",
    "灯塔看守人点亮了灯。",
    "रखवाले ने दीया जलाया।",
    "tabs\tand\nnewlines\r\n",
    "for boat in harbor: print(boat.name)",
]


def random_unicode_text(generator: random.Random, length: int) -> str:
    """Random code points from all planes, skipping surrogates (which UTF-8 cannot encode)."""
    characters = []
    while len(characters) < length:
        code_point = generator.randrange(0x110000)
        if not 0xD800 <= code_point <= 0xDFFF:
            characters.append(chr(code_point))
    return "".join(characters)


@pytest.mark.parametrize("text", TRICKY_TEXTS)
def test_byte_tokenizer_round_trips_tricky_text(text):
    assert ByteTokenizer().round_trips(text)


def test_byte_tokenizer_round_trips_random_unicode():
    generator = random.Random(0)
    tokenizer = ByteTokenizer()
    for _ in range(200):
        assert tokenizer.round_trips(random_unicode_text(generator, 20))


def test_byte_ids_are_utf8_bytes_and_in_range():
    ids = ByteTokenizer().encode("é🐟")
    assert ids == [195, 169, 240, 159, 144, 159]
    assert all(0 <= i < 256 for i in ids)


def test_byte_decode_of_partial_character_uses_replacement():
    assert ByteTokenizer().decode([67, 240, 159]) == "C�"


def test_char_tokenizer_round_trips_text_it_was_trained_on():
    text = "The keeper lit the lamp.\n"
    tokenizer = CharTokenizer.train(text)
    assert tokenizer.round_trips(text)
    assert tokenizer.vocab_size == len(set(text)) + 1


def test_char_tokenizer_maps_unseen_characters_to_unknown():
    tokenizer = CharTokenizer.train("abc")
    assert tokenizer.encode("abz") == [1, 2, 0]
    assert tokenizer.decode(tokenizer.encode("abz")) == "ab" + REPLACEMENT
    assert not tokenizer.round_trips("abz")


def test_char_ids_are_stable_regardless_of_text_order():
    assert CharTokenizer.train("cab").characters == CharTokenizer.train("abc").characters


@pytest.mark.parametrize("tokenizer", [ByteTokenizer(), CharTokenizer.train("harbor lamp")])
def test_save_and_load(tmp_path, tokenizer):
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    loaded = load_tokenizer(path)
    assert type(loaded) is type(tokenizer)
    assert loaded.encode("harbor lamp!") == tokenizer.encode("harbor lamp!")


def test_load_rejects_other_files(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(path)
