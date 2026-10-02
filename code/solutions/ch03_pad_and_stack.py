"""Chapter 3, Exercise 5 (suggested solution): batch sequences of different lengths.

`torch.stack` needs equal shapes, but sentences have different numbers of tokens.
The standard fix is to pad short sequences with a filler ID, and return a mask
recording which positions are real. Chapter 11 builds on this.

Run from `code/`:  python -m solutions.ch03_pad_and_stack
"""

from __future__ import annotations

import torch


def pad_and_stack(sequences: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ids, mask), both of shape (batch, longest length).

    ids[i, j]  is token j of sequence i, or pad_id where sequence i has ended.
    mask[i, j] is True where ids[i, j] is a real token, False where it is padding.
    """
    if not sequences:
        raise ValueError("need at least one sequence")
    longest = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), longest), pad_id, dtype=torch.int64)
    mask = torch.zeros((len(sequences), longest), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.int64)
        mask[row, : len(sequence)] = True
    return ids, mask


def main() -> None:
    ids, mask = pad_and_stack([[5, 9, 2, 5, 7], [5, 1], [8, 9, 2]], pad_id=0)
    print("ids:\n", ids)
    print("mask:\n", mask)
    print("real tokens per sequence:", mask.sum(dim=1).tolist())


if __name__ == "__main__":
    main()
