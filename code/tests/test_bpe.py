"""Tests for the byte-level BPE tokenizer (Chapter 9, Project 1)."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from llmfp.tokenizers import BPETokenizer, load_tokenizer, save_tokenizer
from llmfp.tokenizers.bpe import merge_pair, pretokenize, train_merges, train_merges_reference
from tests.test_tokenizers import TRICKY_TEXTS, random_unicode_text

HARBOR = Path(__file__).resolve().parents[1] / "data" / "tiny" / "harbor.txt"
SAMPLE = HARBOR.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(SAMPLE, vocab_size=300, special_tokens=["<|endoftext|>"])


def test_merge_pair_is_left_to_right_and_non_overlapping():
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]


def test_pretokenize_covers_every_character():
    generator = random.Random(0)
    for _ in range(200):
        text = random_unicode_text(generator, 30)
        assert "".join(pretokenize(text)) == text


def test_pretokenize_keeps_leading_space_with_word():
    assert pretokenize("the keeper's lamp,  lit") == ["the", " keeper", "'s", " lamp", ",", " ", " lit"]


def test_worked_example_first_merges():
    """The merges printed by examples/ch09/bpe_by_hand.py."""
    chunks = Counter(pretokenize("the keeper lit the lamp. the lamp lit the keeper."))
    merges = train_merges(chunks, 6)
    l, h, e, t, space, k, a, m = (ord(c) for c in "lhet kam")
    assert merges == [(space, l), (h, e), (t, 257), (space, 258), (space, k), (a, m)]


@pytest.mark.parametrize("seed", range(5))
def test_fast_training_matches_reference(seed):
    generator = random.Random(seed)
    words = ["lamp", "lamps", "keeper", "keep", "harbor", "the", "lit", "fog", "fish"]
    text = " ".join(generator.choice(words) for _ in range(300))
    chunks = Counter(pretokenize(text))
    assert train_merges(chunks, 40) == train_merges_reference(chunks, 40)


def test_vocab_size_accounting(tokenizer):
    assert tokenizer.vocab_size == 300
    assert len(tokenizer.merges) == 300 - 256 - 1
    assert tokenizer.special_tokens == {"<|endoftext|>": 299}


def test_training_stops_early_when_no_pair_repeats():
    small = BPETokenizer.train(SAMPLE, vocab_size=5000)
    assert small.vocab_size < 5000   # 40 sentences run out of pairs that occur at least twice


@pytest.mark.parametrize("text", TRICKY_TEXTS + [SAMPLE])
def test_round_trip_tricky_text(tokenizer, text):
    assert tokenizer.round_trips(text)


def test_round_trip_random_unicode(tokenizer):
    generator = random.Random(1)
    for _ in range(200):
        assert tokenizer.round_trips(random_unicode_text(generator, 25))


def test_bpe_compresses_text_like_its_training_data(tokenizer):
    # Even 43 merges cut the token count well below one token per byte.
    assert len(tokenizer.encode(SAMPLE)) < len(SAMPLE.encode("utf-8")) * 0.6


def test_special_token_text_is_ordinary_text_by_default(tokenizer):
    text = "end <|endoftext|> here"
    ids = tokenizer.encode(text)
    assert 299 not in ids
    assert tokenizer.decode(ids) == text


def test_special_token_only_when_allowed(tokenizer):
    ids = tokenizer.encode("a<|endoftext|>b", allowed_special={"<|endoftext|>"})
    assert ids.count(299) == 1
    assert tokenizer.decode(ids) == "a<|endoftext|>b"


def test_save_load_round_trip(tmp_path, tokenizer):
    path = tmp_path / "bpe.json"
    save_tokenizer(tokenizer, path)
    loaded = load_tokenizer(path)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode(SAMPLE) == tokenizer.encode(SAMPLE)
    assert loaded.special_tokens == tokenizer.special_tokens


def test_training_is_deterministic():
    first = BPETokenizer.train(SAMPLE, 320)
    second = BPETokenizer.train(SAMPLE, 320)
    assert first.merges == second.merges


def test_vocab_size_too_small_rejected():
    with pytest.raises(ValueError):
        BPETokenizer.train(SAMPLE, 100)


def test_committed_project_tokenizer_loads_and_round_trips():
    path = Path(__file__).resolve().parents[1] / "data" / "tokenizer" / "harbor-bpe-2048.json"
    tokenizer = load_tokenizer(path)
    assert tokenizer.vocab_size == 2048
    assert tokenizer.round_trips("The keeper lit the lamp. Смотритель 🐟\n    def f(x): return x")
