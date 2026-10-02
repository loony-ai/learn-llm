"""Batching variable-length sequences: padding, masks, ignored targets, packing (Chapter 11.6-11.7)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# PyTorch's cross-entropy skips any target equal to this value (its default ignore_index).
IGNORE_INDEX = -100


@dataclass
class PaddedBatch:
    input_ids: torch.Tensor        # (batch, length) int64, padded with pad_id
    targets: torch.Tensor          # (batch, length) int64, IGNORE_INDEX wherever there is no real target
    attention_mask: torch.Tensor   # (batch, length) bool, True at real (non-padding) input positions


def pad_batch(sequences: list[list[int]], pad_id: int, side: str = "right") -> PaddedBatch:
    """Turn token sequences of different lengths into one padded batch of (input, target) pairs.

    Each sequence of n tokens gives n - 1 input/target pairs (the shift by one).
    Padding goes on the right by default; "left" puts it before the tokens, which
    generation code prefers (Chapter 17).
    """
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")
    if any(len(sequence) < 2 for sequence in sequences):
        raise ValueError("every sequence needs at least 2 tokens to form an (input, target) pair")
    length = max(len(sequence) for sequence in sequences) - 1
    batch = len(sequences)
    input_ids = torch.full((batch, length), pad_id, dtype=torch.int64)
    targets = torch.full((batch, length), IGNORE_INDEX, dtype=torch.int64)
    attention_mask = torch.zeros((batch, length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        n = len(sequence) - 1
        where = slice(0, n) if side == "right" else slice(length - n, length)
        input_ids[row, where] = torch.tensor(sequence[:-1])
        targets[row, where] = torch.tensor(sequence[1:])
        attention_mask[row, where] = True
    return PaddedBatch(input_ids, targets, attention_mask)


def pack_documents(
    documents: list[list[int]], context_length: int, separator_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join documents into one stream, separated by `separator_id`, and cut it into windows.

    Returns (windows, document_ids), both (number of windows, context_length + 1):
    each row holds context_length inputs plus one extra token for the last target, and
    document_ids records which document each token came from (separators belong to the
    document they end). Consecutive windows share one token, exactly as
    TokenWindowDataset does with its default stride, so every next-token step in the
    stream is trained once. A leftover tail too short for a full window is dropped.
    """
    stream: list[int] = []
    owners: list[int] = []
    for number, document in enumerate(documents):
        stream.extend(document + [separator_id])
        owners.extend([number] * (len(document) + 1))
    if len(stream) < context_length + 1:
        raise ValueError(f"need at least {context_length + 1} tokens in total, got {len(stream)}")
    width, step = context_length + 1, context_length
    windows = torch.tensor(stream, dtype=torch.int64).unfold(0, width, step)
    document_ids = torch.tensor(owners, dtype=torch.int64).unfold(0, width, step)
    return windows.contiguous(), document_ids.contiguous()
