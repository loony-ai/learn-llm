"""Turning token IDs into training examples and batches (Chapter 11).

    windows.py   TokenWindowDataset: fixed-length (input, target) windows with a stride
    collate.py   padding with masks and ignored targets; packing documents with separators
"""

from llmfp.data.collate import IGNORE_INDEX, PaddedBatch, pack_documents, pad_batch
from llmfp.data.windows import TokenWindowDataset

__all__ = ["IGNORE_INDEX", "PaddedBatch", "TokenWindowDataset", "pack_documents", "pad_batch"]
