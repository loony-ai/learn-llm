"""Tokenizers: turn text into integer IDs and back (Chapters 8-9).

    Tokenizer        common interface (base.py)
    CharTokenizer    one ID per character seen in training text, plus <unk> (char.py)
    ByteTokenizer    one ID per UTF-8 byte value: always 256 IDs, nothing unknown (byte.py)
    BPETokenizer     byte-level byte-pair encoding (bpe.py, Chapter 9)

    save_tokenizer / load_tokenizer   JSON files that record which kind they hold
"""

from llmfp.tokenizers.base import Tokenizer, load_tokenizer, register, save_tokenizer
from llmfp.tokenizers.bpe import BPETokenizer
from llmfp.tokenizers.byte import ByteTokenizer
from llmfp.tokenizers.char import CharTokenizer

__all__ = ["Tokenizer", "CharTokenizer", "ByteTokenizer", "BPETokenizer", "save_tokenizer", "load_tokenizer", "register"]
